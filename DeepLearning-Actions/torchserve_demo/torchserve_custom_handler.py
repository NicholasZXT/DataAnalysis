import torch
from ts.torch_handler.base_handler import BaseHandler

class MyHandler(BaseHandler):
    # 下面3个方法，基本都是 BaseHandler 的实现，只是稍微做了些修改
    def preprocess(self, data):
        # print(data)
        # 打印结果为： [{'body': {'data': [1, 2, 3]}}]
        # print(type(data[0]))
        # 打印结果为： <class 'dict'>
        body = data[0]['body']
        # print(body)
        request_data = body['data']
        # print("request_data: ", request_data)
        # 下面需要使用 expand 转成 shape=(1,3) 的 tensor
        return torch.as_tensor(request_data, device=self.device, dtype=torch.float).expand((1, -1))

    def inference(self, data, *args, **kwargs):
        print("inference get data: ", data)
        with torch.no_grad():
            marshalled_data = data.to(self.device)
            results = self.model(marshalled_data, *args, **kwargs)
        return results

    def postprocess(self, data):
        return data.tolist()