import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTInference:
    def __init__(self, engine_path):
        # TensorRT ロガーの設定
        self.logger = trt.Logger(trt.Logger.INFO)
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
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # 入力の場合
            if self.engine.binding_is_input(binding):
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.outputs.append({"host": host_mem, "device": device_mem})
            self.bindings.append(int(device_mem))

        # ストリームの作成
        self.stream = cuda.Stream()

    def infer(self, input_image):
        # 入力データをホストバッファにコピー
        np.copyto(self.inputs[0]["host"], input_image.ravel())
        # 入力をデバイスに転送
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        # 推論実行
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # 出力をホストに転送
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        # ストリーム同期
        self.stream.synchronize()
        # 出力結果を返す（必要に応じて形状を変更）
        return [out["host"] for out in self.outputs]
