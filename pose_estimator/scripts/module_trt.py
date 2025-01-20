import tensorrt as trt
import cupy as cp
import numpy as np
import rospy

class TensorRTInference:
    def __init__(self, engine_path):
        # TensorRT ロガーの設定
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        # エンジンのデシリアライズ
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # コンテキストの作成
        self.context = self.engine.create_execution_context()

        # 入出力のバッファ設定
        self.bindings = []
        self.inputs = []
        self.outputs = []

        for binding in self.engine:
            # バインディング形状とサイズを取得
            shape = self.engine.get_binding_shape(binding)
            # 動的シェイプの場合、形状が未定義の可能性があるため注意
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # メモリ割り当てサイズ（バイト単位）を計算
            alloc_size = size * np.dtype(dtype).itemsize

            if self.engine.binding_is_input(binding):
                host_mem_ptr = cp.cuda.alloc_pinned_memory(alloc_size)
                device_mem = cp.cuda.memory.alloc(alloc_size)
                # 割り当てたピン留めメモリを NumPy 配列として扱う
                host_mem = np.frombuffer(host_mem_ptr, dtype=dtype, count=size)
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                host_mem_ptr = cp.cuda.alloc_pinned_memory(alloc_size)
                device_mem = cp.cuda.memory.alloc(alloc_size)
                host_mem = np.frombuffer(host_mem_ptr, dtype=dtype, count=size)
                self.outputs.append({"host": host_mem, "device": device_mem})
            self.bindings.append(int(device_mem))

        # CuPy ストリームの作成
        self.stream = cp.cuda.Stream()

    def infer(self, input_image):
        np.copyto(self.inputs[0]["host"], input_image.ravel())
        cp.cuda.runtime.memcpyAsync(
            int(self.inputs[0]["device"]),                 # デバイスポインタを整数に変換
            self.inputs[0]["host"].ctypes.data,            # ホストメモリポインタ
            self.inputs[0]["host"].nbytes,                 
            cp.cuda.runtime.memcpyHostToDevice,
            int(self.stream.ptr))                          # ストリームポインタを整数に変換
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=int(self.stream.ptr))            # ストリームポインタを整数に変換
        for i, out in enumerate(self.outputs):
            cp.cuda.runtime.memcpyAsync(
                self.outputs[i]["host"].ctypes.data,       # ホストメモリポインタ
                int(out["device"]),                        # デバイスポインタを整数に変換
                self.outputs[i]["host"].nbytes,
                cp.cuda.runtime.memcpyDeviceToHost,
                int(self.stream.ptr))                     # ストリームポインタを整数に変換
        self.stream.synchronize()
        return [out["host"] for out in self.outputs]