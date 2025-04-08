import asyncio
import time
import uuid
import os
import logging # 添加 logging
import sys
from enum import Enum
from typing import Dict, List, Optional, Union, Any, AsyncGenerator

from .pipeline import CachedTTSPipeline
import torch
import torchaudio

# 设置日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"      # 等待处理
    PROCESSING = "processing"  # 正在处理
    COMPLETED = "completed"   # 完成
    CANCELED = "canceled"    # 已取消
    FAILED = "failed"        # 失败

class TTSTask:
    def __init__(self, task_id: str, text: str, voice_id: str, speed: float = 1.0, user_id: Optional[str] = None):
        self.task_id = task_id
        self.text = text
        self.voice_id = voice_id
        self.speed = speed
        self.user_id = user_id
        self.status = TaskStatus.PENDING
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.audio_chunks = []  # 存储生成的音频片段

class TTSServiceManager:
    def __init__(
        self, 
        repo_id: str, 
        voices_dir: str, 
        device: Optional[str] = None,
        batch_size: int = 4, 
        max_wait_time: float = 0.2,
        chunk_size: int = 200,
        output_dir: Optional[str] = None  # 新增可选输出目录
    ):
        # 初始化TTS流水线
        self.pipeline = CachedTTSPipeline(repo_id, voices_dir, device)
        
        # 批处理配置
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.chunk_size = chunk_size
        
        # 任务管理
        self.tasks = {}  # 所有任务字典：task_id -> TTSTask
        self.task_queue = asyncio.Queue()  # 待处理任务队列
        
        # 启动批处理循环
        self.processing_task = asyncio.create_task(self._batch_processing_loop())
        
        self.output_dir = output_dir
        
    async def submit_task(self, text: str, voice_id: str, speed: float = 1.0, user_id: Optional[str] = None) -> str:
        """提交一个TTS任务
        
        Args:
            text: 要合成的文本
            voice_id: 语音ID
            speed: 语速
            user_id: 用户ID（可选）
            
        Returns:
            任务ID
        """
        logger.debug(f"提交任务: '{text[:20]}...' voice={voice_id}")
        task_id = str(uuid.uuid4())
        task = TTSTask(task_id, text, voice_id, speed, user_id)
        self.tasks[task_id] = task
        
        # 将任务添加到队列
        logger.debug(f"将任务 {task_id} 添加到队列，当前队列大小: {self.task_queue.qsize()}")
        await self.task_queue.put(task)
        logger.debug(f"任务 {task_id} 已添加到队列")
        
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消一个待处理的任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELED
            return True
        
        # 已经开始处理的任务无法取消
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        if task_id not in self.tasks:
            return None
            
        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "text": task.text,
            "voice_id": task.voice_id,
            "user_id": task.user_id,
            "chunks_completed": len(task.audio_chunks)
        }
    
    async def get_user_tasks(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有任务
        
        Args:
            user_id: 用户ID
            
        Returns:
            任务状态列表
        """
        user_tasks = []
        for task_id, task in self.tasks.items():
            if task.user_id == user_id:
                user_tasks.append(await self.get_task_status(task_id))
        
        return user_tasks
    
    async def stream_result(self, task_id: str) -> AsyncGenerator[torch.Tensor, None]:
        """流式获取任务结果
        
        Args:
            task_id: 任务ID
            
        Yields:
            生成的音频片段
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在: {task_id}")
            
        task = self.tasks[task_id]
        
        # 已完成的任务可以直接返回所有结果
        if task.status == TaskStatus.COMPLETED:
            for chunk in task.audio_chunks:
                yield chunk
            return
        
        # 处理中的任务需要等待新的结果
        current_index = 0
        while task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
            # 如果有新的音频片段
            if current_index < len(task.audio_chunks):
                yield task.audio_chunks[current_index]
                current_index += 1
            else:
                # 等待新的结果
                await asyncio.sleep(0.1)
        
        # 任务完成后，确保返回所有剩余的结果
        while current_index < len(task.audio_chunks):
            yield task.audio_chunks[current_index]
            current_index += 1
    
    def start(self): # 添加 start 方法以在 fixture 外部启动循环
        if not hasattr(self, 'processing_task') or self.processing_task.done():
            logger.info("开始批处理循环...")
            self.processing_task = asyncio.create_task(self._batch_processing_loop())
            logger.info("批处理循环已启动")
        else:
            logger.warning("批处理循环已在运行")

    async def _batch_processing_loop(self):
        """批处理循环"""
        logger.info("批处理循环进入主循环")
        loop_count = 0
        
        while True:
            loop_count += 1
            logger.debug(f"批处理迭代 #{loop_count}")
            batch_tasks = []
            first_task = None
            try:
                # 获取第一个任务，设置较短超时以快速响应
                logger.debug(f"等待第一个任务... (队列大小: {self.task_queue.qsize()})")
                first_task = await asyncio.wait_for(self.task_queue.get(), timeout=self.max_wait_time * 2) # 使用超时
                logger.debug(f"获取到第一个任务: {first_task.task_id}")

                if first_task.status != TaskStatus.PENDING:
                    logger.warning(f"跳过非挂起任务 {first_task.task_id}，状态: {first_task.status}")
                    self.task_queue.task_done()
                    continue

                batch_tasks.append(first_task)
                first_task.status = TaskStatus.PROCESSING # 标记为处理中
                first_task.started_at = time.time()
                logger.debug(f"任务 {first_task.task_id} 状态更新为处理中")

                # 收集更多任务直到达到批次大小或超时
                batch_collection_start_time = time.time()
                while len(batch_tasks) < self.batch_size:
                    try:
                        # 尝试立即获取，不阻塞
                        logger.debug("尝试获取更多任务...")
                        task = self.task_queue.get_nowait()
                        if task.status == TaskStatus.PENDING:
                            batch_tasks.append(task)
                            task.status = TaskStatus.PROCESSING
                            task.started_at = time.time()
                            logger.debug(f"添加任务 {task.task_id} 到批次")
                        else:
                             logger.warning(f"跳过非挂起任务 {task.task_id}，状态: {task.status}")
                             self.task_queue.task_done() # 也要标记完成

                        # 检查是否超时
                        if time.time() - batch_collection_start_time > self.max_wait_time:
                             logger.debug("批次收集已达到最大等待时间")
                             break

                    except asyncio.QueueEmpty:
                        # 如果队列空了，并且等待时间超过 max_wait_time，则处理当前批次
                        if time.time() - batch_collection_start_time > self.max_wait_time:
                            logger.debug("队列为空且已达到最大等待时间")
                            break
                        # 否则短暂等待后继续尝试
                        logger.debug("队列为空，短暂等待...")
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        logger.error(f"获取任务时出错: {e}", exc_info=True)
                        # 避免死循环
                        if time.time() - batch_collection_start_time > self.max_wait_time * 2:
                            logger.warning("由于持续错误，中断批次收集")
                            break
                        await asyncio.sleep(0.05)

            except asyncio.TimeoutError:
                # 等待第一个任务超时，正常现象，继续循环
                logger.debug("等待第一个任务超时，继续循环")
                continue
            except asyncio.CancelledError:
                 logger.info("批处理循环被取消")
                 # 将仍在处理中的任务放回队列或标记为失败？ TBD
                 # 目前简单退出
                 break
            except Exception as e:
                logger.error(f"获取第一个任务时出错: {e}", exc_info=True)
                if first_task: # 如果是因为第一个任务出错，标记它
                    first_task.status = TaskStatus.FAILED
                    first_task.error = str(e)
                    first_task.completed_at = time.time()
                    self.task_queue.task_done() # 别忘了标记
                await asyncio.sleep(0.1) # 发生错误时稍作等待
                continue

            # 如果没有收集到任务，继续循环
            if not batch_tasks:
                logger.debug("没有收集到任务，继续循环")
                continue

            # 处理批次
            logger.info(f"处理 {len(batch_tasks)} 个任务的批次: {[t.task_id for t in batch_tasks]}")
            
            try:
                # 简化这部分：不分片处理，直接调用批处理
                texts = [task.text for task in batch_tasks]
                voice_ids = [task.voice_id for task in batch_tasks]
                speeds = [task.speed for task in batch_tasks]
                
                logger.debug(f"调用批处理: {len(texts)} 个文本")
                # 添加超时，防止无限等待
                try:
                    chunk_results = await asyncio.wait_for(
                        self.pipeline.batch_process_texts(texts, voice_ids, speeds),
                        timeout=10.0
                    )
                    
                    # 处理结果
                    for i, task in enumerate(batch_tasks):
                        audio_chunk = chunk_results[i]
                        task.audio_chunks.append(audio_chunk)
                        logger.debug(f"任务 {task.task_id} 生成了音频块")
                        
                        # 标记任务完成
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = time.time()
                        logger.info(f"任务 {task.task_id} 已完成")
                        
                except asyncio.TimeoutError:
                    logger.error("批处理超时")
                    # 将所有任务标记为失败
                    for task in batch_tasks:
                        task.status = TaskStatus.FAILED
                        task.error = "处理超时"
                        task.completed_at = time.time()
                        logger.error(f"任务 {task.task_id} 因超时而失败")
                        
                except Exception as e:
                    logger.error(f"批处理中发生错误: {e}", exc_info=True)
                    # 将所有任务标记为失败
                    for task in batch_tasks:
                        task.status = TaskStatus.FAILED
                        task.error = f"处理错误: {e}"
                        task.completed_at = time.time()
                        logger.error(f"任务 {task.task_id} 因错误而失败: {e}")
                    
            except Exception as e:
                logger.error(f"处理批次时发生错误: {e}", exc_info=True)
                # 将批次中所有仍在处理的任务标记为失败
                for task in batch_tasks:
                    if task.status == TaskStatus.PROCESSING:
                        task.status = TaskStatus.FAILED
                        task.error = f"批处理错误: {e}"
                        task.completed_at = time.time()
                        logger.error(f"任务 {task.task_id} 因批处理错误而失败: {e}")

            finally:
                # 标记所有任务为完成（无论成功或失败）
                for task in batch_tasks:
                     self.task_queue.task_done()
                logger.info(f"已完成处理 {len(batch_tasks)} 个任务的批次")

    async def save_audio_chunk(self, audio_tensor, filepath, sample_rate):
        """异步保存音频片段到文件"""
        try:
            # 确保张量在CPU上且为浮点类型
            audio_tensor = audio_tensor.cpu().float()
            # torchaudio.save 需要 [C, T] 或 [T] 格式，确保是2D
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.ndim > 2:
                 # 如果有批次维度，取第一个？或者需要调用者处理好
                 logger.warning(f"Audio tensor has unexpected dimensions {audio_tensor.shape} for saving. Taking first element.")
                 audio_tensor = audio_tensor[0].unsqueeze(0)

            await asyncio.to_thread(torchaudio.save, filepath, audio_tensor, sample_rate)
            logger.info(f"Audio chunk saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save audio chunk to {filepath}: {e}", exc_info=True)

    async def shutdown(self):
        """优雅地关闭管理器"""
        logger.info("Shutting down TTSServiceManager...")
        # 停止接受新任务（可以通过应用层逻辑实现）

        # 取消后台处理任务
        if hasattr(self, 'processing_task') and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                logger.info("Batch processing task cancelled.")

        # 等待队列中的任务被处理或标记为取消/失败？（可选）
        # 目前不等待，直接结束
        logger.info("TTSServiceManager shutdown complete.")