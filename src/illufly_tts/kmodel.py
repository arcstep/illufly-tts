from kokoro.model import KModel
import torch

class BatchKModel(KModel):
    @torch.no_grad()
    def forward_batch(self, phonemes_batch, ref_s_batch, speeds=None):
        """批量处理多个语音生成请求
        
        Args:
            phonemes_batch: 批量音素序列 [B, T]
            ref_s_batch: 批量参考语音嵌入 [B, D]
            speeds: 语速列表 [B]
            
        Returns:
            批量音频输出 [B, L]
        """
        if speeds is None:
            speeds = [1.0] * len(phonemes_batch)
        
        # 准备批量输入
        batch_size = len(phonemes_batch)
        
        # 将所有音素转换为input_ids
        input_ids_list = []
        for phonemes in phonemes_batch:
            ids = [self.vocab.get(p) for p in phonemes]
            ids = [id for id in ids if id is not None]
            # 添加开始和结束标记
            input_ids = [0] + ids + [0]
            input_ids_list.append(torch.LongTensor(input_ids))
        
        # 处理不同长度序列（填充到最大长度）
        max_len = max(len(ids) for ids in input_ids_list)
        padded_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, ids in enumerate(input_ids_list):
            length = len(ids)
            padded_ids[i, :length] = ids
            input_lengths[i] = length
        
        # 转移到GPU
        padded_ids = padded_ids.to(self.device)
        input_lengths = input_lengths.to(self.device)
        ref_s_batch = ref_s_batch.to(self.device)
        
        # 创建文本掩码
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(batch_size, -1).to(self.device)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1))
        
        # --- 下面是对现有前向传播的批量改造 ---
        
        # BERT编码（支持批量）
        bert_dur = self.bert(padded_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        
        # 风格和持续时间预测（批量处理）
        s_batch = ref_s_batch[:, 128:]
        d_batch = self.predictor.text_encoder(d_en, s_batch, input_lengths, text_mask)
        x_batch, _ = self.predictor.lstm(d_batch)
        
        # 批量处理持续时间预测
        duration_batch = self.predictor.duration_proj(x_batch)
        duration_batch = torch.sigmoid(duration_batch).sum(axis=-1)
        
        # 应用不同语速
        for i, speed in enumerate(speeds):
            duration_batch[i] = duration_batch[i] / speed
        
        # 后续处理和音频生成（可能需要循环处理）
        audio_outputs = []
        
        for i in range(batch_size):
            # 提取单个样本数据
            curr_len = input_lengths[i]
            curr_padded_ids = padded_ids[i, :curr_len].unsqueeze(0)
            curr_duration = duration_batch[i, :curr_len]
            
            # 处理单个样本的持续时间和对齐
            pred_dur = torch.round(curr_duration).clamp(min=1).long()
            indices = torch.repeat_interleave(
                torch.arange(curr_len, device=self.device), 
                pred_dur
            )
            
            pred_aln_trg = torch.zeros((curr_len, indices.shape[0]), device=self.device)
            pred_aln_trg[torch.arange(curr_len, device=self.device)[:, None], indices] = 1
            pred_aln_trg = pred_aln_trg.unsqueeze(0)
            
            # 提取当前样本编码
            curr_d = d_batch[i].unsqueeze(0)
            curr_s = s_batch[i].unsqueeze(0)
            
            # 计算特征
            en = curr_d.transpose(-1, -2) @ pred_aln_trg
            F0_pred, N_pred = self.predictor.F0Ntrain(en, curr_s)
            
            # 文本编码
            curr_input_length = torch.tensor([curr_len], device=self.device)
            curr_text_mask = text_mask[i, :curr_len].unsqueeze(0)
            t_en = self.text_encoder(curr_padded_ids, curr_input_length, curr_text_mask)
            asr = t_en @ pred_aln_trg
            
            # 解码生成音频
            curr_ref_s = ref_s_batch[i, :128].unsqueeze(0)
            audio = self.decoder(asr, F0_pred, N_pred, curr_ref_s).squeeze()
            audio_outputs.append(audio)
        
        return audio_outputs