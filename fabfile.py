import os
import re
import time

from fabric.api import env
from fabric.colors import blue, red
from fabric.context_managers import cd, hide
from fabric.contrib import files
from fabric.contrib.files import exists
from fabric.decorators import hosts, with_settings
from fabric.operations import run

from fabvenv import virtualenv

ROOT_DIR = '/home/dipath/.dipath_source'
PROJECT_NAME = 'cyborg'
UAT_NAME = f'{PROJECT_NAME}_uat'
PROJECT_GITLAB_URL = f'git@gitlab.dipath.cn:dev/{PROJECT_NAME}.git'
SETTINGS_GITLAB_URL = f'git@gitlab.dipath.cn:dev/technis.git'
online_flag_file = '/home/dipath/status/status.online'
offline_flag_file = '/home/dipath/status/status.offline'

env.user = 'dipath'
env.use_ssh_config = True

SOURCE_DIR = '.'
APP_DIR = '/data/www/cyborg'
VENV_PATH = '/data/venvs/cyborg/'

ALFRED_PROCESS_GROUP = f'{PROJECT_NAME}:'

ONLINE_TAG_NAME = 'online'

NEW_PKG_RE = re.compile('{\+(.*?)\+}')
REMOVED_PKG_RE = re.compile('\[-(.*?)-\]$')


# host: 远端服务器外网IP
# name: 服务器 hostname
# lan: 服务器内网IP

BACKGROUND_SERVERS = []

GOTHAM_HOST = '172.16.1.116'
DEPLOY_HOST = '172.16.1.116'

STAGING_SERVERS = [
    {
        'host': GOTHAM_HOST,
        'name': 'gotham',
        'lan': GOTHAM_HOST,
    },
]

SUPERVISOR = 'sudo supervisorctl'

PROD_SERVERS = [
    {
        'host': GOTHAM_HOST,
        'name': 'gotham',
        'lan': GOTHAM_HOST,
    },
]


@hosts(DEPLOY_HOST)
@with_settings(warn_only=True)
def make_source(commit=None, is_uat=False):
    commit = commit or 'master'
    print(blue('make_source: commit: %s;' % commit))
    if not exists(ROOT_DIR):
        run('mkdir %s' % ROOT_DIR)
    source_dir = os.path.join(ROOT_DIR, UAT_NAME if is_uat else PROJECT_NAME)
    technis_dir = os.path.join(ROOT_DIR, 'technis')

    if not exists(source_dir):
        run('git clone %s %s' % (PROJECT_GITLAB_URL, source_dir))
    with cd(source_dir):
        run('git fetch origin -p')
        run('git checkout -B online origin/%s' % commit)
        deploy_commit = run('git rev-parse HEAD')

    if not exists(technis_dir):
        run('git clone %s %s' % (SETTINGS_GITLAB_URL, technis_dir))
    with cd(technis_dir):
        run('git fetch origin -p')
        run('git checkout -B online origin/main')

    run(f'cp -r {technis_dir}/{PROJECT_NAME}/local_settings {source_dir}/')

    return deploy_commit


@hosts(DEPLOY_HOST)
def deploy(commit=None, check=False):
    env.host_string = DEPLOY_HOST
    deploy_commit = make_source(commit)

    print(blue('线上Web服务器'))
    for server in PROD_SERVERS:
        print(red('\t'.join([server['name'], server['host'], server['lan']])))

    for idx, server in enumerate(PROD_SERVERS):
        remote, server_name = server['host'], server['name']
        sync_source(remote)
        check_server = idx == 0 or check
        success = restart_app(remote, server_name, check=check_server)
        if not success:
            break

    env.host_string = DEPLOY_HOST


def online_fix(commit=None, check=False):
    return deploy(commit=commit, check=check)


@hosts(DEPLOY_HOST)
@with_settings(warn_only=True)
def uat():
    env.host_string = DEPLOY_HOST
    make_source('pre_release', is_uat=True)

    for idx, server in enumerate(STAGING_SERVERS):
        remote, server_name = server['host'], server[
            'name']
        sync_source(remote, is_uat=True)
        with virtualenv(VENV_PATH):
            run('pip install -r {}/requirements.txt'.format(APP_DIR))
        success = restart_app(remote, server_name, check=False)
        if not success:
            break


def restart_app(remote, server_name, check=False):

    mode = '服务器'
    print(red('*' * 30 + ' 上线服务器: %s, 上线粒度:%s ' % (
        server_name, mode) + '*' * 30))

    env.host_string = remote

    # server_offline(remote)

    restart_server(remote)

    with hide('warnings'):
        run('%s status' % SUPERVISOR)

    print(blue('*' * 30 + ' 部署完成: %s ' % server_name + '*' * 30))
    if check:
        input_value = ''
        while input_value not in ('Y', 'N'):
            input_value = input(
                "请输入\"Y\"将服务器上线, (如果输入\"N\", 上线流程将终止)").strip()[
                :1].upper()
            if input_value == 'Y':
                server_online(remote)
            elif input_value == 'N':
                return
        input("请按确认键继续")
    else:
        # server_online(remote)
        print("重启完成, 上线服务器: %s\n" % server_name)

    # 等待负载均衡将服务器上线
    # time.sleep(10)
    return True


def install(package, version=None, is_uat=False):
    if version:
        package = '{0}=={1}'.format(package, version)
    if is_uat:
        for server in STAGING_SERVERS:
            remote = server['host']
            env.host_string = remote
            with virtualenv('/home/lukou/venv/lukou/'):
                run('pip install %s' % package)
    else:
        for server in PROD_SERVERS:
            remote = server['host']
            env.host_string = remote
            with virtualenv('/home/lukou/venv/lukou/'):
                run('pip install %s' % package)

        for server in BACKGROUND_SERVERS:
            remote = server['host']
            env.host_string = remote
            with virtualenv('/home/lukou/venv/lukou/'):
                run('pip install %s' % package)


def server_offline(remote):
    env.host_string = remote
    if files.exists(online_flag_file):
        run('mv %s %s' % (online_flag_file, offline_flag_file))
    elif not files.exists(offline_flag_file):
        run('touch %s' % offline_flag_file)


def server_online(remote):
    env.host_string = remote
    if not files.exists(online_flag_file):
        if files.exists(offline_flag_file):
            run('mv %s %s' % (offline_flag_file, online_flag_file))
        else:
            run('touch %s' % online_flag_file)


def restart_server(remote):
    env.host_string = remote
    # 等待负载均衡下线服务器，2秒 * 3次
    # time.sleep(10)
    with hide('stdout'):
        run('%s restart %s' % (SUPERVISOR, ALFRED_PROCESS_GROUP))
    # 等待服务进程重启完毕(普通启动15秒，newrelic启动24秒)
    time.sleep(5)


def sync_source(remote, deploy_host=DEPLOY_HOST, is_uat=False):
    env.host_string = deploy_host
    source_dir = os.path.join(ROOT_DIR, UAT_NAME if is_uat else PROJECT_NAME)
    with cd(source_dir):
        sync_str = 'rsync -e "ssh -o StrictHostKeyChecking=no" -r ' \
                   '--exclude-from rsync-exclude.txt --delete-after  %s/ ' \
                   '%s@%s:%s'
        run(sync_str % (source_dir, env.user, remote, APP_DIR))
    clear_pyc(remote)


def clear_pyc(remote, coupon_dir=APP_DIR):
    env.host_string = remote
    with cd(coupon_dir):
        run('find . -name \*.pyc -delete')


def run_command(command):
    env.host_string = DEPLOY_HOST
    for host in PROD_SERVERS:
        env.host_string = host['host']
        run(command)


@hosts(DEPLOY_HOST)
def get_new_packages(online_commit, deploy_commit, is_uat=False):
    command = 'git --no-pager diff --word-diff=plain %s %s requirements.txt' % (online_commit, deploy_commit)
    source_dir = os.path.join(ROOT_DIR, UAT_NAME if is_uat else PROJECT_NAME)
    with cd(source_dir):
        result = run(command)
        new_packages = NEW_PKG_RE.findall(result)
        removed_packages = REMOVED_PKG_RE.findall(result)
    return filter(None, set(new_packages) - set(removed_packages))
