[![pipeline status](http://code.idiaoyan.cn/wj/da/seal/badges/pre_release/pipeline.svg)](http://code.idiaoyan.cn/wj/da/seal/-/commits/pre_release)
[![coverage report](http://code.idiaoyan.cn/wj/da/seal/badges/pre_release/coverage.svg)](http://code.idiaoyan.cn/wj/da/seal/-/commits/pre_release)

# seal

数据分析平台代码

* Python3.8.11 + tornado6.1 + MySQL8.0
* Restful API
* 

包含 `algorithm`、`server` `admin` 三个核心部分, 分别对应:
- 项目核心算法包
- 提供给客户端服务的 WebServer
- 提供给后台管理的 WebServer

开发时可以直接使用 `python app.py --port=8888` 运行本地 Server。

## 依赖

使用 [pip-tools](https://github.com/jazzband/pip-tools/) 管理依赖。

安装静态依赖（版本变化需要告知运维升级
```
pip-compile
pip install requirements.txt
```
安装动态依赖（即版本经常变化的依赖，这种依赖每次发布都会自动安装升级
```
pip install requirements-dynamic.txt
```

添加、删除、升级依赖时，首先修改原始的 `requirements.in` 文件（间接依赖不必写入这个文件），然后运行 `pip-compile` 更新 `requirements.txt`，最后通过 `pip-sync` 或者 `pip install -r requirements.txt` 更新虚拟环境。

# 代码规范
- 代码风格参考 [PEP8](https://www.python.org/dev/peps/pep-0008/) ,部分如下:
  - 缩进使用 4个 `space`；（尽量使用单引号吧，目前整个项目都是单引号:-）
  - 包引用
    - 优先标准库
    - 其次相关第三方包
    - 最后是应用的内部包
    - 包引用顺序按字母升序
    - 尽量避免 _wildcard imports_，即 `import *`
  - [`doc string`](https://www.python.org/dev/peps/pep-0008/#documentation-strings) 使用三引号
  - 尽量添加[函数注解](https://www.python.org/dev/peps/pep-0008/#function-annotations)
  > 项目根目录执行: `flake8 && pylint seal` 进行校验
- 版本发布遵循 [Semantic Versioning](https://semver.org/#semantic-versioning-specification-semver)

# shell 调试
- 可通过根目录执行： python shell.py 可进入项目交互环境
- 提供项目内的包环境，自动补全，提示。
- 例如: `from seal.core import cache`

# 数据库与迁移

数据库`ORM`使用 `Sqlalchemy`, 数据库迁移使用 `alembic`.

`ORM` 使用方式参考官方文档: [ORM官方指南](https://docs.sqlalchemy.org/en/14/orm/tutorial.html)

`alembic` 详情参考[官方文档](https://alembic.sqlalchemy.org/en/latest/tutorial.html#tutorial), 常用命令如下:
- `alembic revision --autogenerate -m 'some description'` - 生成最新的数据库迁移文件
- `alembic upgrade [revision|head]` - 一键将迁移文件应用到数据库, revision 表示迁移指定版本，head表示迁移所有到最新
- `alembic downgrade [revision]` - 数据库版本回滚，指定 `revision` 可回滚到指定版本
- `alembic history [-r[start:end]]`  查看版本记录， 可根据revision指定区间

## Q&A

- **Linux** `mysqlclient` 安装失败
```shell
> sudo apt-get install python3-dev default-libmysqlclient-dev build-essential # Debian/Ubuntu
> sudo yum install python3-devel mysql-devel # Red Hat/CentOS
```
- **Mac M1** `mysqlclient` 安装失败
```shell
> brew install mysql  # 或者 mysql-client
> export LDFLAGS="-L/opt/homebrew/opt/mysql-client/lib"  # 对应的安装路径
> export CPPFLAGS="-I/opt/homebrew/opt/mysql-client/include"
```
- **Mac M1** `scikit-learn` 问题，参考[官方安装文档](https://scikit-learn.org/stable/install.html#installing-on-apple-silicon-m1-hardware)
- **Mac M1** `scipy` 问题， 参考[Stackoverflow方法](https://stackoverflow.com/a/66536896)
