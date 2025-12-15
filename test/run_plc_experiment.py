import os
import numpy as np
import sys
import random

def main():
    # ================= 配置区域 =================
    NUM_DROPS = 50             # 随机丢包的次数
    DROP_DURATION_SAMPLES = 240 # 每次丢包的长度 (samples)
    
    # 注意：lpcnet_demo 的处理单元是 packet (320 samples / 20ms)
    # 如果丢包范围跨越了 packet 边界，或者落在 packet 内部，
    # lpcnet_demo 通常会把整个受影响的 packet 视为丢失。
    # 脚本会将任何受影响的 packet 在 pattern 文件中标记为 1。
    
    INPUT_PCM = "See_You_Again_16K_16b_1ch.raw" 
    #INPUT_PCM = "sweep_16k_1ch.raw" 
    OUTPUT_PCM = "output_recovered.pcm"
    PLC_FILE = "plc_pattern.txt"
    OUTPUT_DROPPED_PCM = "output_dropped.pcm"

    # ===========================================

    # 1. 准备输入音频
    if not os.path.exists(INPUT_PCM):
        print(f"Generating synthetic input: {INPUT_PCM}")
        fs = 16000
        duration = 5.0 # seconds
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        # 生成一个 440Hz 的正弦波
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) 
        audio_int16 = (audio * 32767).astype(np.int16)
        with open(INPUT_PCM, "wb") as f:
            f.write(audio_int16.tobytes())
    else:
        print(f"Using existing input: {INPUT_PCM}")

    # 2. 计算 PLC 模式
    # lpcnet_demo 逻辑：每 2 帧 (320 samples) 读取一次 plc_file 中的值
    PACKET_SIZE = 320
    
    file_size = os.path.getsize(INPUT_PCM)
    total_samples = file_size // 2 # 16-bit audio
    total_packets = (total_samples + PACKET_SIZE - 1) // PACKET_SIZE

    print(f"Total samples: {total_samples}")
    print(f"Total packets (320 samples each): {total_packets}")

    # 随机生成丢包位置
    # 避免在开头和结尾太近的地方丢包，以便有上下文
    possible_start_range = (PACKET_SIZE, total_samples - DROP_DURATION_SAMPLES - PACKET_SIZE)
    
    if possible_start_range[1] <= possible_start_range[0]:
        print("Audio too short for drops.")
        return

    drop_starts = []
    for _ in range(NUM_DROPS):
        start = random.randint(possible_start_range[0], possible_start_range[1])
        drop_starts.append(start)
    
    drop_starts.sort()
    print(f"Random drop start samples: {drop_starts}")
    print(f"Drop duration per drop: {DROP_DURATION_SAMPLES} samples")

    # 3. 生成 PLC 文件 和 丢包后的音频
    # 读取原始音频数据
    with open(INPUT_PCM, "rb") as f:
        audio_data = np.frombuffer(f.read(), dtype=np.int16).copy()
    
    # 初始化 pattern 数组 (0: 正常, 1: 丢失)
    pattern = np.zeros(total_packets, dtype=int)
    
    # 标记受影响的 packets 并将音频置零
    # 只要一个 packet 内有任何 sample 被丢弃，该 packet 就标记为丢失
    
    for start_sample in drop_starts:
        end_sample = start_sample + DROP_DURATION_SAMPLES
        
        # 将音频数据置零
        # 注意边界检查
        s_idx = max(0, start_sample)
        e_idx = min(len(audio_data), end_sample)
        audio_data[s_idx:e_idx] = 0
        
        # 计算受影响的 packets
        start_packet = start_sample // PACKET_SIZE
        end_packet = (end_sample + PACKET_SIZE - 1) // PACKET_SIZE
        
        # 标记 pattern
        if end_packet > start_packet:
            pattern[start_packet:end_packet] = 1

    # 写入 PLC 文件
    with open(PLC_FILE, "w") as f:
        for p in pattern:
            f.write(f"{p}\n")
    
    # 保存丢包后的音频
    with open(OUTPUT_DROPPED_PCM, "wb") as f:
        f.write(audio_data.tobytes())
        
    print(f"Generated PLC file: {PLC_FILE}")
    print(f"Generated audio with drops (zeros): {OUTPUT_DROPPED_PCM}")
    
    count_lost = np.sum(pattern)
    print(f"Total packets marked as lost: {count_lost} / {total_packets}")

    print("Experiment setup complete.")
    print(f"1. PLC pattern file: {PLC_FILE}")
    print(f"2. Audio with zeros: {OUTPUT_DROPPED_PCM}")
    print(f"To run LPCNet PLC manually:")
    print(f"lpcnet_demo -plc_file causal {PLC_FILE} {OUTPUT_DROPPED_PCM} {OUTPUT_PCM}")

if __name__ == "__main__":
    main()
