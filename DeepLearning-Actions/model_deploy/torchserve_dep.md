
+ 打包模型

```shell
cd model_deploy  # cd 到指定目录

torch-model-archiver -f --version 1.0
	--model-name mymodel        # 模型的名称，这个名称后续也会作为访问的URL的一部分，打包的文件名称为 mymodel.mar
	--model-file torchserve_dep_model.py   # 模型本身的定义文件，里面只能有一个模型
	--serialized-file D:\Downloads\myModel.pth  # 导出的模型权重文件
	--export-path D:\Downloads\torchserve_models_store  # torchserve 的模型存储仓库，所有的模型都要放到这里才能被注册
	--handler torchserve_custom_handler.py  # 指定自定义的 handler 文件
```

+ 启动服务

```shell
torchserve --start --ncs --model-store D:\Downloads\torchserve_models_store --models mymodel.mar
```

+ 访问服务

```shell
# 查看torchserve后端服务的情况
curl --location --request GET '127.0.0.1:8081/ping'

# 查看已经注册的模型
curl --location --request GET '127.0.0.1:8081/models'

# 查看可以访问的URL
curl --location --request OPTIONS '127.0.0.1:8080'

# 请求mymodel的服务
curl --location --request GET '127.0.0.1:8080/predictions/mymodel' \
	--header 'Content-Type: application/json' \
    --data-raw '{
        "data": [1,2,3]
    }'
```