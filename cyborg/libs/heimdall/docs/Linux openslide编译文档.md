## 1. github拉取代码

-   通过`git clone https://github.com/openslide/openslide.git`拉取最新版代码
-   Ventana支持相关代码在以下仓库中`git clone https://github.com/Path-AI/openslide.git -b dp200-support-with-tests`
-   将Ventana代码直接替换openslide/src 下对应文件就可以。

## 2. 修改Aperio相关

-   在src下修改`openslide-vendor-aperio.c`,在大约490行左右找到 `//associated image` 注释，将下方代码修改为以下代码,为加载label相关代码：
```c
// associated image
const char *name = (dir == 1) ? "thumbnail" : NULL;
if (!add_associated_image(osr, name, tc, ct.tiff, err)) {
return false;
}
const char *name1 = (dir == level_array->len+2) ? "label" :NULL;
if (!add_associated_image(osr, name1, tc, ct.tiff, err)){
return false;
}
//g_debug("associated image: %d", dir);
```

## 3. 安装编译所需依赖（Ubuntu 环境下）
-   安装编译所需环境，包括 libtiff-4,libjpeg,libpng,glib-2.0,gdk-pixbuf-2.0,cairo,libxml,sqlite3,zlib,meson,gcc,make,cmake
```bash
sudo apt-get update #更新apt index list
sudo apt-get install libtiff-dev libjpeg-dev libpng-dev libglib2.0-dev libgdk-pixbuf2.0-dev libcairo2-dev libxml2-dev libsqlite3-dev zlib1g-dev
sudo apt-get install meson gcc make cmake
```

此外，还需要openjpeg 库，需要从官网下载openjpeg源码进行编译。openjpeg 2.5.0 下载链接`https://github.com/uclouvain/openjpeg/releases/download/v2.5.0/openjpeg-v2.5.0-linux-x86_64.tar.gz`,解压后将文件夹放到openslide源码文件夹下。

## 3. 进行编译
1.  在openslide外一级文件夹执行`meson setup openslide`
2.  执行`meson compile -C ./`
3.  执行`sudo meson install -C ./` 即可编译完成，编译完的so库位于openslide外一级src下