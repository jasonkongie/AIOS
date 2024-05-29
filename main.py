import os
import sys
import json

from src.scheduler.fifo_scheduler import FIFOScheduler
from src.scheduler.rr_scheduler import RRScheduler
from src.utils.utils import parse_global_args
from openagi.src.agents.agent_factory import AgentFactory
from openagi.src.agents.agent_process import AgentProcessFactory
import warnings
from src.llm_kernel import llms
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.utils import delete_directories
from dotenv import find_dotenv, load_dotenv

def clean_cache(root_directory):
    targets = {'.ipynb_checkpoints', '__pycache__', ".pytest_cache", "context_restoration"}
    delete_directories(root_directory, targets)

def main():
    warnings.filterwarnings("ignore")
    parser = parse_global_args()
    args = parser.parse_args()

    llm_name = args.llm_name
    max_gpu_memory = args.max_gpu_memory
    eval_device = args.eval_device
    max_new_tokens = args.max_new_tokens
    scheduler_log_mode = args.scheduler_log_mode
    agent_log_mode = args.agent_log_mode
    llm_kernel_log_mode = args.llm_kernel_log_mode
    load_dotenv()

    llm = llms.LLMKernel(
        llm_name=llm_name,
        max_gpu_memory=max_gpu_memory,
        eval_device=eval_device,
        max_new_tokens=max_new_tokens,
        log_mode=llm_kernel_log_mode
    )

    scheduler = RRScheduler(
        llm=llm,
        log_mode=scheduler_log_mode
    )

    agent_process_factory = AgentProcessFactory()

    agent_factory = AgentFactory(
        llm=llm,
        agent_process_queue=scheduler.agent_process_queue,
        agent_process_factory=agent_process_factory,
        agent_log_mode=agent_log_mode
    )

    agent_thread_pool = ThreadPoolExecutor(max_workers=64)

    scheduler.start()

    # construct agents
    streaming_mode_agent = agent_thread_pool.submit(
        agent_factory.run_agent,
        "StreamingModeAgent",
        "Manage video streaming modes based on memory availability and operational requirements."
    )

    disk_storage_agent = agent_thread_pool.submit(
        agent_factory.run_agent,
        "DiskStorageAgent",
        "Manage disk space to ensure there is enough space for video storage and other system operations."
    )

    agent_tasks = [streaming_mode_agent, disk_storage_agent]

    for r in as_completed(agent_tasks):
        res = r.result()

    scheduler.stop()

    clean_cache(root_directory="./")

if __name__ == "__main__":
    main()
