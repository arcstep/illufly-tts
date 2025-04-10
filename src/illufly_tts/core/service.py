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
        self.debug_id = None  # 新增debug_id属性

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
        
        self.output_dir = output_dir
        self.processing_task = None  # 初始化时不创建任务
        
    async def submit_task(self, text: str, voice_id: str, speed: float = 1.0, user_id: Optional[str] = None, debug_id: Optional[str] = None) -> str:
        """提交一个TTS任务
        
        Args:
            text: 要合成的文本
            voice_id: 语音ID
            speed: 语速
            user_id: 用户ID（可选）
            debug_id: 调试ID（可选）
            
        Returns:
            任务ID
        """
        logger.debug(f"提交任务: '{text[:20]}...' voice={voice_id}")
        
        # 先验证语音是否存在
        try:
            # 尝试预加载语音验证其存在性
            await asyncio.to_thread(self.pipeline.load_voice, voice_id)
        except ValueError as e:
            logger.error(f"语音加载失败（严重错误）: {e}")
            # 创建失败任务并立即标记失败
            task_id = str(uuid.uuid4())
            task = TTSTask(task_id, text, voice_id, speed, user_id)
            task.status = TaskStatus.FAILED
            task.error = f"语音加载错误: {str(e)}"
            task.completed_at = time.time()
            self.tasks[task_id] = task
            return task_id
        
        task_id = str(uuid.uuid4())
        task = TTSTask(task_id, text, voice_id, speed, user_id)
        task.debug_id = debug_id  # 添加这行
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
        """流式获取任务结果"""
        if task_id not in self.tasks:
            logger.error(f"找不到任务: {task_id}")
            raise ValueError(f"找不到任务: {task_id}")
        
        task = self.tasks[task_id]
        
        # 等待任务完成
        while task.status == TaskStatus.PROCESSING:
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.FAILED:
            logger.error(f"任务失败: {task_id}, 错误: {task.error}")
            raise ValueError(f"任务失败: {task.error}")
        
        if task.status == TaskStatus.CANCELED:
            logger.error(f"任务被取消: {task_id}")
            raise ValueError("任务被取消")
        
        # 流式返回音频块
        for chunk in task.audio_chunks:
            logger.info(f"返回音频块: {task_id}, 形状: {chunk.shape}")
            
            # 调试输出
            debug_output_dir = os.environ.get("TTS_DEBUG_OUTPUT")
            if debug_output_dir and hasattr(task, 'debug_id') and task.debug_id:
                # 创建递增的文件名
                chunk_index = task.audio_chunks.index(chunk)
                debug_path = os.path.join(debug_output_dir, f"{task.debug_id}_stream_chunk_{chunk_index}.wav")
                try:
                    logger.error(f"调试(stream): 尝试保存音频块到 {debug_path}")
                    # 保存文件
                    audio_to_save = chunk.cpu().float()
                    if audio_to_save.ndim == 1:
                        audio_to_save = audio_to_save.unsqueeze(0)
                    await asyncio.to_thread(torchaudio.save, debug_path, audio_to_save, 24000)
                    logger.error(f"调试(stream): 已保存音频块到 {debug_path}")
                except Exception as e:
                    logger.error(f"调试(stream): 保存音频块失败: {e}")
            
            yield chunk

    async def start(self):
        """异步启动服务"""
        if not self.processing_task or self.processing_task.done():
            logger.info("开始批处理循环...")
            self.processing_task = asyncio.create_task(self._batch_processing_loop())
            logger.info("批处理循环已启动")
        else:
            logger.warning("批处理循环已在运行")

    async def _batch_processing_loop(self):
        """批处理循环"""
        while True:
            # 收集待处理的任务
            pending_tasks = [task for task in self.tasks.values() 
                              if task.status == TaskStatus.PENDING]
            
            if not pending_tasks:
                await asyncio.sleep(0.1)
                continue
            
            # 等待足够的任务或超过最大等待时间
            if len(pending_tasks) < self.batch_size:
                start_time = time.time()
                while (len(pending_tasks) < self.batch_size and 
                       time.time() - start_time < self.max_wait_time):
                    await asyncio.sleep(0.05)
                    # 更新待处理任务列表
                    pending_tasks = [task for task in self.tasks.values() 
                                    if task.status == TaskStatus.PENDING]
                    if len(pending_tasks) >= self.batch_size:
                        break
            
            # 选取批次大小或全部任务
            batch_tasks = pending_tasks[:self.batch_size]
            logger.info(f"处理批次: {len(batch_tasks)} 任务")
            
            # 更新任务状态
            for task in batch_tasks:
                task.status = TaskStatus.PROCESSING
            
            try:
                # 收集所有文本
                batch_texts = [task.text for task in batch_tasks]
                # 收集所有语音ID
                batch_voice_ids = [task.voice_id for task in batch_tasks]
                # 收集所有速度设置
                batch_speeds = [task.speed for task in batch_tasks]
                
                # 批量处理
                chunk_results = await self._async_batch_process(
                    batch_texts, batch_voice_ids, batch_speeds)
                
                # 处理结果
                for i, task in enumerate(batch_tasks):
                    audio_chunk = chunk_results[i]
                    
                    # 调试输出
                    debug_output_dir = os.environ.get("TTS_DEBUG_OUTPUT")
                    if debug_output_dir and hasattr(task, 'debug_id') and task.debug_id:
                        debug_path = os.path.join(debug_output_dir, f"{task.debug_id}_pipeline_output.wav")
                        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                        try:
                            logger.error(f"调试(pipeline): 尝试保存输出到 {debug_path}")
                            # 保存文件
                            audio_to_save = audio_chunk.cpu().float()
                            if audio_to_save.ndim == 1:
                                audio_to_save = audio_to_save.unsqueeze(0)
                            await asyncio.to_thread(torchaudio.save, debug_path, audio_to_save, 24000)
                            logger.error(f"调试(pipeline): 已保存管道输出到 {debug_path}")
                        except Exception as e:
                            logger.error(f"调试(pipeline): 保存音频失败: {e}")
                    
                    task.audio_chunks.append(audio_chunk)
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    logger.info(f"任务完成: {task.task_id}, 文本长度: {len(task.text)}")
                    
                    # 如果配置了输出目录，保存音频
                    if self.output_dir:
                        output_path = os.path.join(self.output_dir, f"{task.task_id}.wav")
                        await self.save_audio_chunk(audio_chunk, output_path, 24000)
                        logger.info(f"保存音频到: {output_path}")
            
            except Exception as e:
                logger.error(f"批处理失败: {e}", exc_info=True)
                # 更新所有任务状态为失败
                for task in batch_tasks:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()

    async def _async_batch_process(self, texts, voice_ids, speeds=None):
        """异步包装器，用于调用pipeline的批处理方法"""
        logger.info(f"调用异步批处理: {len(texts)}个文本")
        # 如果pipeline有异步方法，优先使用
        if hasattr(self.pipeline, 'async_batch_process_texts'):
            return await self.pipeline.async_batch_process_texts(texts, voice_ids, speeds)
        # 否则，使用同步方法并放在线程池中执行
        else:
            logger.info("使用线程池执行同步批处理方法")
            return await asyncio.to_thread(
                self.pipeline.batch_process_texts,
                texts, voice_ids, speeds
            )

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