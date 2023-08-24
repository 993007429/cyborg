import os
from logging.config import dictConfig

from flask import Flask, Response
from flask.json.provider import DefaultJSONProvider

from cyborg.app.limiter import limiter
from cyborg.app.settings import Settings
from cyborg.app.logging import gen_logging_config
from cyborg.app.subscribe import subscribe_events
from cyborg.utils.encoding import CyborgJsonEncoder


class CustomJSONProvider(DefaultJSONProvider):
    def default(self, o):
        return CyborgJsonEncoder().default(o)


def init_app():
    from cyborg.app.api import api_blueprint, user, record, slice_file, slice_analysis, ai, report, admin # type: ignore

    """项目初始化"""
    # 主应用的根目录
    app = Flask(__name__, static_url_path='',
                static_folder=os.path.join(Settings.PROJECT_DIR, 'static'))

    # 加载配置
    app.config.from_object(Settings)

    # 增加频率限制
    limiter.init_app(app)

    # 启用日志
    dictConfig(gen_logging_config(logging_filename=Settings.APP_LOG_FILE))

    # 输出格式化
    app.json = CustomJSONProvider(app)

    @app.route('/')  # 首页路由
    def index():
        mid_path = os.path.join(Settings.PROJECT_DIR, 'static/index.html')
        return Response(open(mid_path).read(), mimetype="text/html")

    # 注册蓝图 解耦
    app.register_blueprint(api_blueprint, url_prefix='/aipath/api')

    # TODO 按需加载
    from cyborg.app.openapi.roche import roche_blueprint, oauth, vendor  # type: ignore
    app.register_blueprint(roche_blueprint, url_prefix='/roche')

    subscribe_events()

    return app


app = init_app()
