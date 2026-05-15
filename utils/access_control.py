import os
import heapq
import copy
import contextvars
from aiohttp import web
from typing import Optional

import folder_paths
from server import PromptServer
from execution import PromptQueue, MAXIMUM_HISTORY_SIZE

from .users_db import UsersDB


class AccessControl:
    def __init__(self, users_db: UsersDB, server: PromptServer):
        self.users_db = users_db
        self.server = server

        self._current_user = contextvars.ContextVar("user_id", default=None)
        self._executing_user = contextvars.ContextVar("executing_user_id", default=None)
        self.__current_user_id = None
        self.__prompt_user_ids = {}

        self.__get_output_directory = folder_paths.get_output_directory
        self.__get_temp_directory = folder_paths.get_temp_directory
        self.__get_input_directory = folder_paths.get_input_directory

        self.__prompt_queue = self.server.prompt_queue
        self.__prompt_queue_put = self.__prompt_queue.put

    @property
    def folder_paths(self) -> tuple:
        return (
            self.__get_output_directory(),
            self.__get_temp_directory(),
            self.__get_input_directory(),
        )

    def set_current_user_id(self, user_id: str, set_fallback: bool = False) -> None:
        """Set the current user directory from ID."""
        self._current_user.set(user_id)

        if set_fallback:
            self.__current_user_id = user_id

    def get_current_user_id(self) -> str:
        """Retrieve the current user directory from ID."""
        if self._current_user.get():
            return self._current_user.get()

        if self._executing_user.get():
            return self._executing_user.get()

        return self.__current_user_id

    def is_current_user_admin(self) -> bool:
        user_id, user = self.users_db.get_user(user_id=self.get_current_user_id())
        return bool(user_id and user.get("admin"))

    def get_current_user_directory_name(self) -> str:
        """Return the folder name for the current user."""
        user_id = self.get_current_user_id()
        if not user_id:
            return "public"

        _, user = self.users_db.get_user(user_id=user_id)
        return user.get("username") or "public"

    @staticmethod
    def get_prompt_id(item):
        return item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else None

    def is_prompt_visible_to_user(self, item, user_id: str, is_admin: bool) -> bool:
        if is_admin:
            return True

        prompt_id = self.get_prompt_id(item)
        return self.__prompt_user_ids.get(prompt_id) == user_id

    def get_user_output_directory(self) -> str:
        """Get the user-specific output directory."""
        return os.path.join(
            self.__get_output_directory(),
            self.get_current_user_directory_name(),
        )

    def get_user_temp_directory(self) -> str:
        """Get the user-specific temp directory."""
        return os.path.join(
            self.__get_temp_directory(),
            self.get_current_user_directory_name(),
        )

    def get_user_input_directory(self) -> str:
        """Get the user-specific input directory."""
        input_directory = os.path.join(
            self.__get_input_directory(),
            self.get_current_user_directory_name(),
        )

        os.makedirs(input_directory, exist_ok=True)

        return input_directory

    def add_user_specific_folder_paths(self, json_data) -> None:
        """Add user-specific folder paths to the prompt JSON data."""
        directory_name = self.get_current_user_directory_name()

        if isinstance(json_data, dict):
            for key, value in json_data.items():
                if key == "filename_prefix":
                    json_data[key] = f"{directory_name}/{value}"
                else:
                    self.add_user_specific_folder_paths(value)
        elif isinstance(json_data, list):
            for item in json_data:
                self.add_user_specific_folder_paths(item)

        return json_data

    def patch_folder_paths(self) -> None:
        """Patch the folder_paths with user-specific methods."""
        # folder_paths.get_output_directory = self.get_user_output_directory
        folder_paths.get_temp_directory = self.get_user_temp_directory
        folder_paths.get_input_directory = self.get_user_input_directory

        self.server.add_on_prompt_handler(self.add_user_specific_folder_paths)

    def create_folder_access_control_middleware(
        self, folder_paths: tuple = ()
    ) -> web.middleware:
        """Create middleware for folder access control."""

        folder_paths = folder_paths or self.folder_paths

        def get_path_parts(path: str) -> list[str]:
            return [
                part
                for part in path.replace("\\", "/").strip("/").split("/")
                if part
            ]

        def get_requested_folder_user_id(request: web.Request) -> Optional[str]:
            if request.path in {
                "/view",
                "/api/view",
            }:
                file_type = request.query.get("type", "input")
                if file_type not in {"input", "output", "temp"}:
                    return None

                subfolder = request.query.get("subfolder", "")
                filename = request.query.get("filename", "")
                if not subfolder and not filename:
                    return None

                subfolder_parts = get_path_parts(subfolder)
                filename_parts = get_path_parts(filename)
                if ".." in subfolder_parts or ".." in filename_parts:
                    return "__forbidden__"

                if subfolder_parts:
                    return subfolder_parts[0]

                if len(filename_parts) < 2:
                    return None

                return filename_parts[0]

            if request.path.startswith(folder_paths):
                relative_path = request.path
                for folder_path in folder_paths:
                    if request.path.startswith(folder_path):
                        relative_path = os.path.relpath(request.path, folder_path)
                        break

                parts = relative_path.replace("\\", "/").strip("/").split("/", 1)
                return parts[0] if parts and parts[0] else None

            return None

        @web.middleware
        async def folder_access_control_middleware(
            request: web.Request, handler
        ) -> web.Response:
            """Middleware to handle folder access control."""
            folder_user_id = get_requested_folder_user_id(request)

            if not folder_user_id:
                return await handler(request)

            user_id = request.get("user_id")
            user_id, user = self.users_db.get_user(user_id=user_id)

            if folder_user_id == "__forbidden__":
                return web.HTTPForbidden(reason="Path traversal is not allowed.")

            if folder_user_id == "public":
                return await handler(request)

            if (
                not user_id
                or not user
                or (user.get("username") != folder_user_id and not user.get("admin"))
            ):
                return web.HTTPForbidden(
                    reason="You do not have access to this folder."
                )

            return await handler(request)

        return folder_access_control_middleware

    def user_queue_put(self, item):
        """Put an item in the user-specific queue."""
        with self.__prompt_queue.mutex:
            prompt_id = self.get_prompt_id(item)
            if prompt_id is not None:
                self.__prompt_user_ids[prompt_id] = self.get_current_user_id()
            self.__prompt_queue_put(item)

    def user_queue_get(self, timeout=None):
        """Get an item from the user-specific queue."""
        user_queue = self.__prompt_queue.queue
        with self.__prompt_queue.not_empty:
            while len(user_queue) == 0:
                self.__prompt_queue.not_empty.wait(timeout=timeout)
                if timeout is not None and len(user_queue) == 0:
                    return None
            item = heapq.heappop(user_queue)
            prompt_id = self.get_prompt_id(item)
            self._executing_user.set(self.__prompt_user_ids.get(prompt_id))
            i = self.__prompt_queue.task_counter
            self.__prompt_queue.currently_running[i] = copy.deepcopy(item)
            self.__prompt_queue.task_counter += 1
            self.server.queue_updated()
            return (item, i)

    def user_queue_task_done(
        self,
        item_id,
        history_result,
        status: Optional["PromptQueue.ExecutionStatus"],
        process_item=None,
    ):
        """Mark a user-specific queue task as done."""
        with self.__prompt_queue.mutex:
            prompt = self.__prompt_queue.currently_running.pop(item_id)
            if len(self.__prompt_queue.history) > MAXIMUM_HISTORY_SIZE:
                self.__prompt_queue.history.pop(next(iter(self.__prompt_queue.history)))

            status_dict: Optional[dict] = None
            if status is not None:
                status_dict = copy.deepcopy(status._asdict())

            prompt_tuple = prompt
            prompt_id = self.get_prompt_id(prompt_tuple)
            user_id = self.__prompt_user_ids.pop(prompt_id, None)
            if process_item is not None:
                prompt_tuple = process_item(prompt_tuple)
                prompt_id = self.get_prompt_id(prompt_tuple)

            self.__prompt_queue.history[prompt_id] = {
                "prompt": prompt_tuple,
                "outputs": {},
                "status": status_dict,
                "user_id": user_id,
            }
            self.__prompt_queue.history[prompt_id].update(history_result)
            self._executing_user.set(None)
            self.server.queue_updated()

    def user_queue_get_current_queue(self):
        """Get the current user-specific queue."""
        with self.__prompt_queue.mutex:
            current_user_id = self.get_current_user_id()
            is_admin = self.is_current_user_admin()

            out = []
            for x in self.__prompt_queue.currently_running.values():
                if self.is_prompt_visible_to_user(x, current_user_id, is_admin):
                    out.append(x)

            queued = []
            for x in self.__prompt_queue.queue:
                if self.is_prompt_visible_to_user(x, current_user_id, is_admin):
                    queued.append(x)

            return (out, copy.deepcopy(queued))

    def user_queue_get_tasks_remaining(self):
        """Get the number of user-visible remaining queue tasks."""
        running, queued = self.user_queue_get_current_queue()
        return len(running) + len(queued)

    def user_queue_wipe_queue(self):
        """Wipe the user-specific queue."""
        with self.__prompt_queue.mutex:
            current_user_id = self.get_current_user_id()
            is_admin = self.is_current_user_admin()

            if is_admin:
                removed_prompt_ids = [
                    self.get_prompt_id(item) for item in self.__prompt_queue.queue
                ]
                self.__prompt_queue.queue = []
            else:
                removed_prompt_ids = [
                    self.get_prompt_id(item)
                    for item in self.__prompt_queue.queue
                    if self.__prompt_user_ids.get(self.get_prompt_id(item))
                    == current_user_id
                ]
                self.__prompt_queue.queue = [
                    item
                    for item in self.__prompt_queue.queue
                    if self.__prompt_user_ids.get(self.get_prompt_id(item))
                    != current_user_id
                ]
            for prompt_id in removed_prompt_ids:
                self.__prompt_user_ids.pop(prompt_id, None)
            self.server.queue_updated()

    def user_queue_delete_queue_item(self, function):
        """Delete an item from the user-specific queue."""
        with self.__prompt_queue.mutex:
            current_user_id = self.get_current_user_id()
            is_admin = self.is_current_user_admin()

            for x in range(len(self.__prompt_queue.queue)):
                item = self.__prompt_queue.queue[x]
                prompt_id = self.get_prompt_id(item)
                user_id = self.__prompt_user_ids.get(prompt_id)
                if (
                    function(item)
                    and (is_admin or user_id == current_user_id)
                ):
                    if len(self.__prompt_queue.queue) == 1:
                        self.__prompt_queue.wipe_queue()
                    else:
                        self.__prompt_queue.queue.pop(x)
                        heapq.heapify(self.__prompt_queue.queue)
                        self.__prompt_user_ids.pop(prompt_id, None)
                    self.server.queue_updated()
                    return True
        return False

    def user_queue_get_history(
        self, prompt_id=None, max_items=None, offset=-1, map_function=None
    ):
        """Get the user-specific queue history."""
        with self.__prompt_queue.mutex:
            current_user_id = self.get_current_user_id()
            is_admin = self.is_current_user_admin()

            user_history = {
                k: v
                for k, v in self.__prompt_queue.history.items()
                if is_admin
                or v.get("user_id") == current_user_id
            }
            if prompt_id is None:
                out = {}
                i = 0
                if offset < 0 and max_items is not None:
                    offset = len(self.__prompt_queue.history) - max_items
                for k in user_history:
                    if i >= offset:
                        p = user_history[k]
                        if map_function is None:
                            p = copy.deepcopy(p)
                        else:
                            p = map_function(p)
                        out[k] = p
                        if max_items is not None and len(out) >= max_items:
                            break
                    i += 1
                return out
            elif prompt_id in user_history:
                p = user_history[prompt_id]
                if map_function is None:
                    p = copy.deepcopy(p)
                else:
                    p = map_function(p)
                return {prompt_id: p}
            else:
                return {}

    def user_queue_wipe_history(self):
        """Wipe the user-specific queue history."""
        with self.__prompt_queue.mutex:
            current_user_id = self.get_current_user_id()
            is_admin = self.is_current_user_admin()

            if is_admin:
                self.__prompt_queue.history = {}
                return

            self.__prompt_queue.history = {
                k: v
                for k, v in self.__prompt_queue.history.items()
                if v.get("user_id") != current_user_id
            }

    def user_queue_delete_history_item(self, id_to_delete):
        """Delete a user-specific queue history item."""
        with self.__prompt_queue.mutex:
            current_user_id = self.get_current_user_id()
            is_admin = self.is_current_user_admin()

            item = self.__prompt_queue.history.get(id_to_delete)
            if item and (is_admin or item.get("user_id") == current_user_id):
                self.__prompt_queue.history.pop(id_to_delete, None)

    def patch_prompt_queue(self):
        """Patch the prompt queue with user-specific methods."""
        self.__prompt_queue.put = self.user_queue_put
        self.__prompt_queue.get = self.user_queue_get
        self.__prompt_queue.task_done = self.user_queue_task_done
        self.__prompt_queue.get_current_queue = self.user_queue_get_current_queue
        self.__prompt_queue.get_current_queue_volatile = (
            self.user_queue_get_current_queue
        )
        self.__prompt_queue.get_tasks_remaining = self.user_queue_get_tasks_remaining
        self.__prompt_queue.wipe_queue = self.user_queue_wipe_queue
        self.__prompt_queue.delete_queue_item = self.user_queue_delete_queue_item
        self.__prompt_queue.get_history = self.user_queue_get_history
        self.__prompt_queue.wipe_history = self.user_queue_wipe_history
        self.__prompt_queue.delete_history_item = self.user_queue_delete_history_item

    def create_manager_access_control_middleware(
        self, manager_directory: str = "/extensions/comfyui-manager", manager_routes: tuple = ()
    ) -> web.middleware:
        """Create middleware for manager access control."""

        @web.middleware
        async def manager_access_control_middleware(
            request: web.Request, handler
        ) -> web.Response:
            """Middleware to handle manager access control."""
            user_id = request.get("user_id")
            
            if self.users_db.get_admin_user()[0] == user_id or (not request.path.startswith(manager_routes) and not request.path.lower().startswith(manager_directory)):
                return await handler(request)

            return web.HTTPForbidden(
                reason="You do not have access to comfyui manager."
            )

        return manager_access_control_middleware
