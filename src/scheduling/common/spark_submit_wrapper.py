# -*- coding: utf-8 -*-

import os
import sys
import logging
import shell_wrapper

reload(sys)
sys.path.append(os.getcwd())


SPARK_HOME = None
if os.environ.get("SPARK_HOME", "") != "":
    SPARK_HOME = os.environ["SPARK_HOME"].strip()
assert SPARK_HOME, "error! SPARK_HOME not set!"


class SparkSubmitWrapper(object):
    def __init__(self):

        self._app_jar = ""
        self._app_arguments = []

        """
        --master
            spark://host:port, mesos://host:port, yarn,
            k8s://https://host:port, or local (Default: local[*]).
        """
        self._master = ""

        """
        --deploy-mode
            Whether to launch the driver program locally ("client") or 
            on one of the worker machines inside the cluster ("cluster")
            (Default: client).
        """
        self._deploy_model = "client"

        """
        --class
            Your application's main class (for Java / Scala apps).
        """
        self._class = ""

        """
        --name
            A name of your application.
        """
        self._name = ""

        """
        --jars
            list of jars to include on the driver and executor classpaths.
        """
        self._jars = []

        """
        --packages
            list of maven coordinates of jars to include
            on the driver and executor classpaths. Will search the local
            maven repo, then maven central and any additional remote
            repositories given by --repositories. The format for the
            coordinates should be groupId:artifactId:version.
        """
        self._packages = []

        """
        --exclude-packages
            list of groupId:artifactId, to exclude while
            resolving the dependencies provided in --packages to avoid
            dependency conflicts.
        """
        self._exclude_packages = []

        """
        --repositories
            Comma-separated list of additional remote repositories to
            search for the maven coordinates given with --packages.
        """
        self._repositories = []

        """
        --py-files
            list of .zip, .egg, or .py files to place
            on the PYTHONPATH for Python apps.
        """
        self._py_files = []

        """
        --files
            list of files to be placed in the working
            directory of each executor. File paths of these files
            in executors can be accessed via SparkFiles.get(fileName).
        """
        self._files = []

        """
        --conf
            PROP=VALUE. Arbitrary Spark configuration property.
        """
        self._conf = {}

        """
        --properties-file
            Path to a file from which to load extra properties. If not
            specified, this will look for conf/spark-defaults.conf.
        """
        self._properties_file = ""

        """
        --driver-memory
            Memory for driver (e.g. 1000M, 2G) (Default: 1024M).
        """
        self._driver_memory = ""

        """
        --driver-java-options
            Extra Java options to pass to the driver.
        """
        self._driver_java_options = ""

        """
        --driver-library-path
            Extra library path entries to pass to the driver.
        """
        self._driver_library_path = ""

        """
        --driver-class-path
            Extra class path entries to pass to the driver. Note that
            jars added with --jars are automatically included in the classpath.
        """
        self._driver_class_path = ""

        """
        --executor-memory
            Memory per executor (e.g. 1000M, 2G) (Default: 1G).
        """
        self._executor_memory = ""

        """
        --proxy-user
            User to impersonate when submitting the application.
            This argument does not work with --principal / --keytab.
        """
        self._proxy_user = ""

        """
        --driver-cores (Cluster deploy mode only)
            Number of cores used by the driver, only in cluster mode (Default: 1).
        """
        self._driver_cores = -1

        """
        --total-executor-cores (Spark standalone and Mesos only)
            Total cores for all executors.
        """
        self._total_executor_cores = -1

        """
        --executor-cores (Spark standalone and YARN only)
            Number of cores per executor. (Default: 1 in YARN mode,
            or all available cores on the worker in standalone mode)
        """
        self._executor_cores = -1

        """
        --queue (YARN-only)
            The YARN queue to submit to (Default: "default").
        """
        self._queue = ""

        """
        --num-executors (YARN-only)
            Number of executors to launch (Default: 2).
            If dynamic allocation is enabled, the initial number of
            executors will be at least NUM.
        """
        self._num_executors = -1

        """
        --archives (YARN-only)
            list of archives to be extracted into the
            working directory of each executor.
        """
        self._archives = []

        """
        --principal (YARN-only)
            Principal to be used to login to KDC, while running on secure HDFS.
        """
        self._principal = ""

        """
        --keytab (YARN-only)
            The full path to the file that contains the keytab for the
            principal specified above. This keytab will be copied to
            the node running the Application Master via the Secure
            Distributed Cache, for renewing the login tickets and the
            delegation tokens periodically.
        """
        self._keytab = ""

    def set_app_jar(self, jar):
        assert jar != ""
        self._app_jar = jar
        return self

    def add_app_argument(self, arg_key, arg_val=None):
        assert arg_key != ""
        if arg_val is None:
            self._app_arguments.append(arg_key)
        else:
            self._app_arguments.append("{}={}".format(arg_key, arg_val))
        return self

    def set_master(self, master):
        assert master != ""
        self._master = master
        return self

    def set_deploy_mode(self, deploy_mode):
        assert deploy_mode != ""
        self._deploy_model = deploy_mode
        return self

    def set_class(self, param_class):
        assert param_class != ""
        self._class = param_class
        return self

    def set_name(self, name):
        assert name != ""
        self._name = name
        return self

    def add_jar(self, *jars):
        assert len(jars) > 0
        self._jars.extend(jars)
        return self

    def add_package(self, *packages):
        assert len(packages) > 0
        self._packages.extend(packages)
        return self

    def add_exclude_package(self, *packages):
        assert len(packages) > 0
        self._exclude_packages.extend(packages)
        return self

    def add_repository(self, *repositories):
        assert len(repositories) > 0
        self._repositories.extend(repositories)
        return self

    def add_py_file(self, *py_files):
        assert len(py_files) > 0
        self._py_files.extend(py_files)
        return self

    def add_file(self, *files):
        assert len(files) > 0
        self._files.extend(files)
        return self

    def add_conf(self, key, value):
        assert key != ""
        assert value != ""
        self._conf[key] = value
        return self

    def set_properties_file(self, properties_file):
        assert properties_file != ""
        self._properties_file = properties_file
        return self

    def set_driver_memory(self, driver_memory):
        assert isinstance(driver_memory, str)
        assert driver_memory.endswith("M") or driver_memory.endswith("G"), "memory value must end with M/G"
        self._driver_memory = driver_memory
        return self

    def set_driver_java_options(self, driver_java_options):
        assert driver_java_options != ""
        self._driver_java_options = driver_java_options
        return self

    def set_driver_library_path(self, driver_library_path):
        assert driver_library_path != ""
        self._driver_class_path = driver_library_path
        return self

    def set_driver_class_path(self, driver_class_path):
        assert driver_class_path != ""
        self._driver_class_path = driver_class_path
        return self

    def set_executor_memory(self, executor_memory):
        assert isinstance(executor_memory, str)
        assert executor_memory.endswith("M") or executor_memory.endswith("G"), "memory value must end with M/G"
        self._executor_memory = executor_memory
        return self

    def set_proxy_user(self, proxy_user):
        assert proxy_user != ""
        self._proxy_user = proxy_user
        return self

    def set_driver_cores(self, driver_cores):
        assert isinstance(driver_cores, int) and driver_cores > 0
        self._driver_cores = driver_cores
        return self

    def set_total_executor_cores(self, total_executor_cores):
        assert isinstance(total_executor_cores, int)
        assert total_executor_cores > 0
        self._total_executor_cores = total_executor_cores
        return self

    def set_executor_cores(self, executor_cores):
        assert isinstance(executor_cores, int)
        assert executor_cores > 0
        self._executor_cores = executor_cores
        return self

    def set_queue(self, queue):
        assert queue != ""
        self._queue = queue
        return self

    def set_num_executors(self, num_executors):
        assert isinstance(num_executors, int)
        assert num_executors > 0
        self._num_executors = num_executors
        return self

    def add_archive(self, *archives):
        assert len(archives) > 0
        self._archives.extend(archives)
        return self

    def set_principal(self, principal):
        assert principal != ""
        self._principal = principal
        return self

    def set_keytab(self, keytab):
        assert keytab != ""
        self._keytab = keytab
        return self

    def build_cmd(self):
        cmd = "{}/bin/spark-submit".format(SPARK_HOME)
        if self._master:
            cmd += " --master {}".format(self._master)
        if self._deploy_model:
            cmd += " --deploy-mode {}".format(self._deploy_model)
        if self._class:
            cmd += " --class {}".format(self._class)
        if self._name:
            cmd += " --name {}".format(self._name)
        if len(self._jars) > 0:
            cmd += " --jars {}".format(",".join(self._jars))
        if len(self._packages) > 0:
            cmd += " --packages {}".format(",".join(self._packages))
        if len(self._exclude_packages) > 0:
            cmd += " --exclude-packages {}".format(",".join(self._exclude_packages))
        if len(self._repositories) > 0:
            cmd += " --repositories {}".format(",".join(self._repositories))
        if len(self._py_files) > 0:
            cmd += " --py-files {}".format(",".join(self._py_files))
        if len(self._files) > 0:
            cmd += " --files {}".format(",".join(self._files))
        for k, v in self._conf.viewitems():
            cmd += " --conf {}={}".format(str(k), str(v))
        if self._properties_file:
            cmd += " --properties-file {}".format(self._properties_file)
        if self._driver_memory:
            cmd += " --driver-memory {}".format(self._driver_memory)
        if self._driver_java_options:
            cmd += " --driver-java-options {}".format(self._driver_java_options)
        if self._driver_library_path:
            cmd += " --driver-library-path {}".format(self._driver_library_path)
        if self._driver_class_path:
            cmd += " --driver-class-path {}".format(self._driver_class_path)
        if self._executor_memory:
            cmd += " --executor-memory {}".format(self._executor_memory)
        if self._proxy_user:
            cmd += " --proxy-user {}".format(self._proxy_user)
        if self._driver_cores > 0:
            cmd += " --driver-cores {}".format(self._driver_cores)
        if self._total_executor_cores > 0:
            cmd += " --total-executor-cores {}".format(self._total_executor_cores)
        if self._executor_cores > 0:
            cmd += " --executor-cores {}".format(self._executor_cores)
        if self._queue:
            cmd += " --queue {}".format(self._queue)
        if self._num_executors > 0:
            cmd += " --num-executors {}".format(self._num_executors)
        if len(self._archives) > 0:
            cmd += " --archives {}".format(self._archives)
        if self._principal:
            cmd += " --principal {}".format(self._principal)
        if self._keytab:
            cmd += " --keytab {}".format(self._keytab)
        if self._app_jar:
            cmd += " {}".format(self._app_jar)
        if len(self._app_arguments) > 0:
            for arg in self._app_arguments:
                cmd += " {}".format(arg)
        return cmd

    def run(self, **kwargs):
        cmd = self.build_cmd()
        if kwargs.get("print_cmd", False):
            logging.info(cmd)
        return shell_wrapper.shell_command(cmd=cmd, print_info=kwargs.get("print_info", False))


if __name__ == '__main__':
    submit = SparkSubmitWrapper()
    submit.set_master("yarn")
    submit.set_deploy_mode("cluster")
    submit.set_driver_memory("1G")
    submit.set_executor_memory("4G")
    submit.set_executor_cores(2)
    submit.set_num_executors(50)
    submit.set_queue("root.hadoop")
    submit.set_name("test_proc")
    submit.add_conf("spark.network.timeout", "300")
    submit.add_conf("spark.akka.timeout", "300")
    submit.set_class("com.test")
    submit.set_app_jar("xxxx.jar")
    submit.add_app_argument("arg1")
    submit.add_app_argument("arg2")
    print >> sys.stdout, submit.build_cmd()
