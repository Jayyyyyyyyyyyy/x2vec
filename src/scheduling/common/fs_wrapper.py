# -*- coding: utf-8 -*-

import sys
import os
import shutil
import logging

reload(sys)
sys.path.append(os.getcwd())


def exists(path):
    return os.path.exists(path)


def is_file(path):
    return os.path.isfile(path)


def is_dir(path):
    return os.path.isdir(path)


def is_link(path):
    return os.path.islink(path)


def normalize_path(path):
    return os.path.normpath(path)


def full_norm_path(path):
    return normalize_path(os.path.abspath(path))


def dir_path(path):
    return normalize_path(os.path.dirname(os.path.abspath(path)))


def parent_dir(path):
    return normalize_path(os.path.dirname(os.path.abspath(path)) + "/..")


def file_size(path, unit="B"):
    bytes_cnt = os.path.getsize(path)
    if unit == "B":
        pass
    elif unit == "K":
        bytes_cnt = bytes_cnt / 1024.0
    elif unit == "M":
        bytes_cnt = bytes_cnt / 1024.0 / 1024.0
    elif unit == "G":
        bytes_cnt = bytes_cnt / 1024.0 / 1024.0 / 1024.0
    elif unit == "T":
        bytes_cnt = bytes_cnt / 1024.0 / 1024.0 / 1024.0 / 1024.0
    else:
        raise Exception("error unit [{}] for file_size".format(unit))
    return bytes_cnt


def remove_file(*paths):
    for path in paths:
        if not exists(path):
            logging.warning("path [%s] does not exist, skip removing", path)
            continue
        if not is_file(path):
            logging.warning("path [%s] must be a file, exit removing", path)
            return False
        os.remove(path)
        if exists(path):
            logging.error("fail to remove path [%s]", path)
            return False
    return True


def remove_dir(*paths):
    for path in paths:
        if not exists(path):
            logging.warning("path [%s] does not exist, skip removing", path)
            continue
        if dir_path(path).rstrip("/").count("/") < 3:
            raise Exception("must not remove dir close to root")
        shutil.rmtree(path)
        if exists(path):
            return False
    return True


def copy_file(*paths):
    """
    params: src_file_1, src_file_2, ... , dest_path
    1. when only one src file is provided, dest_path can be either a file or a directory
    2. when multiple src_files are provided, dest_path must be an existing directory
    """
    if len(paths) < 2:
        raise Exception("error params for copy_file, at least one src and dest")
    if len(paths) > 2 and (not exists(paths[-1]) or not is_dir(paths[-1])):
        raise Exception("error params for copy_file, dest must be directory when multiple src are provided")

    for path in paths[:-1]:
        try:
            shutil.copy(path, paths[-1])
        except Exception as ex:
            logging.exception("exception occur in %s, %s", __file__, ex)
            return False

    return True


def copy_dir(src, dest):
    """
    src and dest must be both directories and dest should not exist
    """
    try:
        shutil.copytree(src, dest)
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        return False
    return True


def move(*paths):
    """
    Recursively move a file or directory (src) to another location (dst)
    1. If tow params provided,
         if the destination is an existing directory, then src is moved inside that directory.
         if the destination already exists but is not a directory, failed
    2. If multiple params provided, last one should be the existing destination directory.
    """
    paths_cnt = len(paths)
    if paths_cnt < 2:
        raise Exception("error params for move, at least one src and dest")
    elif paths_cnt == 2:
        try:
            shutil.move(paths[0], paths[1])
        except Exception as ex:
            logging.exception("exception occur in %s, %s", __file__, ex)
            return False
    else:
        if not exists(paths[-1]) or not is_dir(paths[-1]):
            raise Exception("error params for move, dest must be an existing directory")
        for path in paths[:-1]:
            try:
                shutil.move(path, paths[-1])
            except Exception as ex:
                logging.exception("exception occur in %s, %s", __file__, ex)
                return False
    return True


def mkdir(*paths):
    for path in paths:
        if exists(path):
            logging.warning("path %s already exists, skip mkdir", path)
            continue
        os.makedirs(path)
        if not exists(path) or not is_dir(path):
            return False
    return True


def md5sum(input_file, output_md5_file):
    if not os.path.exists(input_file):
        logging.warning('file NOT exists: %s', input_file)
        return False

    def read_chunks(fh):
        fh.seek(0)
        chunk = fh.read(8096)
        while chunk:
            yield chunk
            chunk = fh.read(8096)
        else:
            fh.seek(0)

    import hashlib
    m = hashlib.md5()
    with open(input_file, "rb") as fh:
        for chunk in read_chunks(fh):
            m.update(chunk)
    md5_str = m.hexdigest()
    if not md5_str:
        return False
    with open(output_md5_file, "w") as fout:
        print >> fout, "{}  {}".format(md5_str, os.path.basename(input_file))
    return exists(output_md5_file)


def is_same_md5(first_md5_file, second_md5_file):
    """compare md5 string
    """
    try:
        first_md5_str = open(first_md5_file, 'r').readline().split(' ')[0]
        second_md5_str = open(second_md5_file, 'r').readline().split(' ')[0]
        return first_md5_str == second_md5_str
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
    return False


def gzip_compress(dest_pack, *src_files, **kwargs):
    if len(src_files) == 0:
        raise Exception("must provide at least one src file for gzip compressing")
    import tarfile
    with tarfile.open(dest_pack, "w:gz") as tar:
        working_dir = kwargs.get("working_dir", "")
        for src_file in src_files:
            try:
                if working_dir != "":
                    working_dir = full_norm_path(working_dir)
                    src_file = full_norm_path(src_file)
                    if not src_file.startswith(working_dir):
                        logging.error("src path does not start with specified working_dir [%s]", working_dir)
                        return False
                    arcname = src_file[len(working_dir):].lstrip("/")
                    if arcname == "":
                        arcname = "."
                    tar.add(src_file, arcname=arcname)
                else:
                    tar.add(src_file)
            except Exception as ex:
                logging.exception("exception occur in %s, %s", __file__, ex)
                return False
    return exists(dest_pack)


def gzip_decompress(src_pack, dest_dir, filter_indicator=None):
    import tarfile
    try:
        with tarfile.open(src_pack, "r:gz") as tar:
            if dest_dir is None:
                dest_dir = "."
            members = [x for x in tar if filter_indicator(x.name)] if filter_indicator is not None else None
            tar.extractall(path=dest_dir, members=members)
    except Exception as ex:
        logging.exception("exception occur in %s, %s", __file__, ex)
        return False
    return True


def touch(filename):
    dirname = dir_path(filename)
    if not exists(dirname) and not mkdir(dirname):
        logging.error("fail to mkdir %s for touch", dirname)
        return False
    open(filename, "a+").close()
    return exists(filename)


if __name__ == "__main__":
    pass
