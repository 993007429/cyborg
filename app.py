from cyborg.app.init import app
from cyborg.app.settings import Settings


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Settings.PORT, threaded=True)
