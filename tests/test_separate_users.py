import asyncio
import heapq
import importlib
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class HTTPForbidden(Exception):
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason


def install_dependency_stubs():
    aiohttp = types.ModuleType("aiohttp")
    web = types.SimpleNamespace(
        middleware=lambda function: function,
        HTTPForbidden=HTTPForbidden,
        Response=object,
    )
    aiohttp.web = web
    sys.modules["aiohttp"] = aiohttp
    sys.modules["aiohttp.web"] = web

    bcrypt = types.ModuleType("bcrypt")
    bcrypt.gensalt = lambda: b"salt"
    bcrypt.hashpw = lambda password, salt: b"hashed"
    bcrypt.checkpw = lambda password, hashed: password == b"password"
    sys.modules["bcrypt"] = bcrypt

    bleach = types.ModuleType("bleach")
    bleach.clean = lambda value, **kwargs: value
    sys.modules["bleach"] = bleach

    jwt = types.ModuleType("jwt")

    class ExpiredSignatureError(Exception):
        pass

    class DecodeError(Exception):
        pass

    class InvalidTokenError(Exception):
        pass

    jwt.ExpiredSignatureError = ExpiredSignatureError
    jwt.DecodeError = DecodeError
    jwt.InvalidTokenError = InvalidTokenError
    jwt.payload = {}
    jwt.encode = lambda data, key, algorithm=None: "token"
    jwt.decode = lambda token, key, algorithms=None: jwt.payload
    sys.modules["jwt"] = jwt

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: "/tmp/comfy-output"
    folder_paths.get_temp_directory = lambda: "/tmp/comfy-temp"
    folder_paths.get_input_directory = lambda: "/tmp/comfy-input"
    sys.modules["folder_paths"] = folder_paths

    server = types.ModuleType("server")

    class PromptServer:
        pass

    server.PromptServer = PromptServer
    sys.modules["server"] = server

    execution = types.ModuleType("execution")
    execution.MAXIMUM_HISTORY_SIZE = 10000

    class PromptQueue:
        class ExecutionStatus(tuple):
            pass

    execution.PromptQueue = PromptQueue
    sys.modules["execution"] = execution


install_dependency_stubs()

from utils.access_control import AccessControl
from utils.config import get_separate_users
from utils.jwt_auth import JWTAuth
from utils.users_db import UsersDB


class FakeQueue:
    def __init__(self):
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.not_empty.notify()


class FakeServer:
    def __init__(self):
        self.prompt_queue = FakeQueue()
        self.updated = 0
        self.prompt_handler = None

    def queue_updated(self):
        self.updated += 1

    def add_on_prompt_handler(self, handler):
        self.prompt_handler = handler


class FakeRequest(dict):
    def __init__(self, path, user_id=None, query=None, headers=None, cookies=None):
        super().__init__()
        self.path = path
        self.query = query or {}
        self.headers = headers or {}
        self.cookies = cookies or {}
        if user_id is not None:
            self["user_id"] = user_id


async def ok_handler(request):
    return "ok"


class SeparateUsersTests(unittest.TestCase):
    def make_users_db(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "users.json"
        db_path.write_text(
            json.dumps(
                {
                    "user-1": {"username": "alice", "password": "hash"},
                    "user-2": {"username": "bob", "password": "hash"},
                    "admin": {
                        "username": "admin",
                        "password": "hash",
                        "admin": True,
                    },
                }
            )
        )
        return UsersDB(db_path)

    def test_separate_users_config_accepts_canonical_and_legacy_keys(self):
        self.assertTrue(get_separate_users({"separate_users": True}))
        self.assertTrue(get_separate_users({"seperate_users": True}))
        self.assertFalse(get_separate_users({}))
        self.assertFalse(
            get_separate_users({"separate_users": False, "seperate_users": True})
        )

    def test_get_user_by_id_returns_two_tuple_and_handles_missing_ids(self):
        users_db = self.make_users_db()

        self.assertEqual(
            users_db.get_user(user_id="user-1"),
            ("user-1", {"username": "alice", "password": "hash"}),
        )
        self.assertEqual(users_db.get_user(user_id="missing"), (None, {}))

    def test_jwt_middleware_sets_fallback_for_prompt_routes(self):
        users_db = self.make_users_db()

        class FakeAccessControl:
            def __init__(self):
                self.calls = []

            def set_current_user_id(self, user_id, set_fallback=False):
                self.calls.append((user_id, set_fallback))

        access_control = FakeAccessControl()
        logger = types.SimpleNamespace(error=lambda message: None)
        auth = JWTAuth(users_db, access_control, logger, "secret")

        jwt = importlib.import_module("jwt")
        jwt.payload = {"id": "user-1", "username": "alice"}
        middleware = auth.create_jwt_middleware()

        for path in ("/prompt", "/api/prompt"):
            request = FakeRequest(path, headers={"Authorization": "Bearer token"})
            result = asyncio.run(middleware(request, ok_handler))
            self.assertEqual(result, "ok")

        self.assertEqual(access_control.calls, [("user-1", True), ("user-1", True)])

    def test_folder_access_middleware_controls_view_subfolders_by_username(self):
        users_db = self.make_users_db()
        access_control = AccessControl(users_db, FakeServer())
        middleware = access_control.create_folder_access_control_middleware()

        allowed = FakeRequest(
            "/view",
            user_id="user-1",
            query={
                "type": "output",
                "filename": "image.png",
                "subfolder": "alice/workflow",
            },
        )
        self.assertEqual(asyncio.run(middleware(allowed, ok_handler)), "ok")

        forbidden = FakeRequest(
            "/view",
            user_id="user-1",
            query={
                "type": "output",
                "filename": "image.png",
                "subfolder": "bob/workflow",
            },
        )
        result = asyncio.run(middleware(forbidden, ok_handler))
        self.assertIsInstance(result, HTTPForbidden)

        public = FakeRequest(
            "/view",
            user_id="user-1",
            query={
                "type": "output",
                "filename": "image.png",
                "subfolder": "public/workflow",
            },
        )
        self.assertEqual(asyncio.run(middleware(public, ok_handler)), "ok")

        unscoped_filename = FakeRequest(
            "/view",
            user_id="user-1",
            query={"type": "output", "filename": "image.png"},
        )
        self.assertEqual(asyncio.run(middleware(unscoped_filename, ok_handler)), "ok")

        traversal = FakeRequest(
            "/view",
            user_id="user-1",
            query={"type": "output", "subfolder": "alice/../bob"},
        )
        result = asyncio.run(middleware(traversal, ok_handler))
        self.assertIsInstance(result, HTTPForbidden)

        filename_traversal = FakeRequest(
            "/view",
            user_id="user-1",
            query={
                "type": "output",
                "subfolder": "alice",
                "filename": "../bob/secret.png",
            },
        )
        result = asyncio.run(middleware(filename_traversal, ok_handler))
        self.assertIsInstance(result, HTTPForbidden)

    def test_folder_access_middleware_allows_admin_to_view_other_user_folder(self):
        access_control = AccessControl(self.make_users_db(), FakeServer())
        middleware = access_control.create_folder_access_control_middleware()
        request = FakeRequest(
            "/view",
            user_id="admin",
            query={"type": "output", "filename": "alice/image.png"},
        )

        self.assertEqual(asyncio.run(middleware(request, ok_handler)), "ok")

    def test_prompt_queue_patch_unwraps_items_and_isolates_history(self):
        access_control = AccessControl(self.make_users_db(), FakeServer())
        queue = access_control.server.prompt_queue
        access_control.patch_prompt_queue()

        access_control.set_current_user_id("user-1")
        queue.put((1, "prompt-1", {}, {}, []))
        queue.put((3, "prompt-3", {}, {}, []))

        access_control.set_current_user_id("user-2")
        queue.put((2, "prompt-2", {}, {}, []))
        self.assertTrue(all(isinstance(item, tuple) for item in queue.queue))

        running, queued = queue.get_current_queue()
        self.assertEqual(running, [])
        self.assertEqual([item[1] for item in queued], ["prompt-2"])
        self.assertEqual(queue.get_tasks_remaining(), 1)

        access_control.set_current_user_id("admin")
        running, queued = queue.get_current_queue()
        self.assertEqual(running, [])
        self.assertEqual(
            sorted(item[1] for item in queued),
            ["prompt-1", "prompt-2", "prompt-3"],
        )

        access_control.set_current_user_id("user-2")
        self.assertTrue(queue.delete_queue_item(lambda item: item[1] == "prompt-2"))
        access_control.set_current_user_id("user-1")
        self.assertEqual(
            sorted(item[1] for item in queue.get_current_queue()[1]),
            ["prompt-1", "prompt-3"],
        )

        access_control.set_current_user_id("user-1")
        prompt, item_id = queue.get()
        self.assertEqual(prompt[1], "prompt-1")
        access_control._current_user.set(None)
        self.assertEqual(access_control.get_current_user_id(), "user-1")
        access_control.set_current_user_id("user-1")

        queue.task_done(
            item_id,
            {"outputs": {"node": {"images": []}}},
            None,
            process_item=lambda item: (item[0], "processed-prompt", item[2], item[3], item[4]),
        )
        history = queue.get_history()
        self.assertEqual(list(history), ["processed-prompt"])
        self.assertEqual(history["processed-prompt"]["user_id"], "user-1")
        access_control._current_user.set(None)
        self.assertIsNone(access_control.get_current_user_id())
        access_control.set_current_user_id("user-1")

        mapped = queue.get_history(map_function=lambda item: item["prompt"][1])
        self.assertEqual(mapped, {"processed-prompt": "processed-prompt"})

        access_control.set_current_user_id("user-2")
        self.assertEqual(queue.get_history(), {})
        queue.delete_history_item("processed-prompt")
        access_control.set_current_user_id("user-1")
        self.assertIn("processed-prompt", queue.get_history())

        queue.delete_history_item("processed-prompt")
        self.assertEqual(queue.get_history(), {})

    def test_admin_can_manage_all_queue_and_history_items(self):
        access_control = AccessControl(self.make_users_db(), FakeServer())
        queue = access_control.server.prompt_queue
        access_control.patch_prompt_queue()

        access_control.set_current_user_id("user-1")
        queue.put((1, "prompt-1", {}, {}, []))
        access_control.set_current_user_id("user-2")
        queue.put((2, "prompt-2", {}, {}, []))

        access_control.set_current_user_id("admin")
        self.assertTrue(queue.delete_queue_item(lambda item: item[1] == "prompt-1"))
        self.assertEqual([item[1] for item in queue.get_current_queue()[1]], ["prompt-2"])

        access_control.set_current_user_id("user-2")
        prompt, item_id = queue.get()
        queue.task_done(item_id, {}, None)

        access_control.set_current_user_id("admin")
        self.assertEqual(list(queue.get_history()), [prompt[1]])
        queue.wipe_history()
        self.assertEqual(queue.get_history(), {})

    def test_user_directories_use_username(self):
        access_control = AccessControl(self.make_users_db(), FakeServer())
        access_control.set_current_user_id("user-1")
        self.assertEqual(
            access_control.get_user_output_directory(),
            "/tmp/comfy-output/alice",
        )
        self.assertEqual(
            access_control.get_user_temp_directory(),
            "/tmp/comfy-temp/alice",
        )

        prompt = {
            "1": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "nested": [{"filename_prefix": "Nested"}],
                }
            }
        }

        access_control.add_user_specific_folder_paths(prompt)

        self.assertEqual(prompt["1"]["inputs"]["filename_prefix"], "alice/ComfyUI")
        self.assertEqual(
            prompt["1"]["inputs"]["nested"][0]["filename_prefix"], "alice/Nested"
        )


if __name__ == "__main__":
    unittest.main()
