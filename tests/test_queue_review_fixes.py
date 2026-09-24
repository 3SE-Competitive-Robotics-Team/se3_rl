"""验证队列调度、资源查询和状态恢复的外部行为。"""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import mujoco

from scripts import check_closedchain_model as checker
from scripts import se3_training_queue as queue


class QueueReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.settings = queue.Settings(root, root, Path("uv"), (0, 1))

    def add_job(self, **fields):
        values = dict(
            status="queued",
            name="验证",
            task="Flat",
            git_commit="test",
            num_envs=1,
            iterations=1,
            save_interval=1,
            smoke=0,
            wandb_mode="disabled",
            extra_args_json="[]",
            extra_env_json="{}",
            created_at=queue._now(),
        )
        values.update(fields)
        with queue._connect(self.settings) as connection:
            cursor = connection.execute(
                f"INSERT INTO jobs ({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})",
                list(values.values()),
            )
            return int(cursor.lastrowid)

    def test_resources_preserves_stale_job(self):
        job_id = self.add_job(status="running", pid=123, assigned_gpu_ids_json="[0]")
        before = dict(queue._job(self.settings, job_id))
        output = io.StringIO()
        with (
            patch.object(queue, "_external_training_pids", return_value=[]),
            patch.object(queue, "_busy_gpu_ids", return_value=set()),
            patch.object(queue, "_process_alive", return_value=False),
            contextlib.redirect_stdout(output),
        ):
            queue._cmd_resources(argparse.Namespace(), self.settings)
        self.assertEqual(dict(queue._job(self.settings, job_id)), before)
        self.assertEqual(json.loads(output.getvalue())["leased_gpu_ids"], [0])

    def test_recovery_preserves_concurrent_completion_and_new_executor(self):
        for initial_status, initial_pid in (("running", 123), ("starting", None)):
            for final_status, final_pid in (("succeeded", 123), ("running", 456)):
                with self.subTest(initial=initial_status, final=final_status):
                    job_id = self.add_job(
                        status=initial_status, pid=initial_pid, assigned_gpu_ids_json="[0]"
                    )

                    def concurrent_update(
                        pid, target=job_id, status=final_status, replacement_pid=final_pid
                    ):
                        queue._update_job(self.settings, target, status=status, pid=replacement_pid)
                        return False

                    with patch.object(queue, "_process_alive", side_effect=concurrent_update):
                        rows = queue._recover_active_jobs(self.settings)
                    row = queue._job(self.settings, job_id)
                    self.assertEqual((row["status"], row["pid"]), (final_status, final_pid))
                    self.assertEqual(
                        job_id in {row["id"] for row in rows}, final_status == "running"
                    )
                    queue._update_job(self.settings, job_id, status="succeeded")

    def test_recovery_requeues_abandoned_start(self):
        job_id = self.add_job(status="starting", assigned_gpu_ids_json="[0]")
        queue._recover_active_jobs(self.settings)
        row = queue._job(self.settings, job_id)
        self.assertEqual(row["status"], "queued")
        self.assertIsNone(row["assigned_gpu_ids_json"])

    def test_gpu_environment_uses_pci_order(self):
        job_id = self.add_job(assigned_gpu_ids_json="[1]")
        with patch.dict(queue.os.environ, CUDA_DEVICE_ORDER="FASTEST_FIRST"):
            environment = queue._clean_environment(self.settings, queue._job(self.settings, job_id))
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
        with self.assertRaises(SystemExit):
            queue._parse_env(["CUDA_DEVICE_ORDER=FASTEST_FIRST"])

    def run_worker_once(self, busy):
        with (
            patch.object(queue, "fcntl", Mock(LOCK_EX=2, LOCK_NB=4)),
            patch.object(queue, "_external_training_pids", return_value=[]),
            patch.object(queue, "_busy_gpu_ids", return_value=busy),
            patch.object(queue, "_launch_job", return_value=Mock(pid=123)) as launch,
            patch.object(queue.time, "sleep", side_effect=InterruptedError),
            self.assertRaises(InterruptedError),
        ):
            queue._cmd_worker(argparse.Namespace(poll_seconds=1), self.settings)
        return launch

    def test_impossible_request_does_not_block_next_job(self):
        oversized = self.add_job(gpu_count=4)
        next_id = self.add_job(gpu_count=1)
        launch = self.run_worker_once(set())
        self.assertEqual(queue._job(self.settings, oversized)["status"], "failed")
        self.assertEqual(launch.call_args.args[1]["id"], next_id)

    def test_temporary_shortage_preserves_queue_order(self):
        first = self.add_job(gpu_count=2)
        second = self.add_job(gpu_count=1)
        launch = self.run_worker_once({0})
        launch.assert_not_called()
        for job_id in (first, second):
            self.assertEqual(queue._job(self.settings, job_id)["status"], "queued")


class ModelValidationTests(unittest.TestCase):
    def test_missing_passive_joint_fails(self):
        model = mujoco.MjModel.from_xml_string("<mujoco/>")
        with self.assertRaisesRegex(SystemExit, "缺少必需的被动关节"):
            checker._check_passive_joint_ranges(model)


if __name__ == "__main__":
    unittest.main()
