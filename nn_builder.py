from __future__ import absolute_import, division, print_function, unicode_literals
import pkgutil
import inspect

from Networks.NNInterface import NNInterface

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~  Objects constructors by name   ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_network(network_type, classes_num, input_size):
    import Networks
    package = Networks
    return get_object(network_type, package, classes_num, input_size)






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~  Construct object by class name   ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_module_classes(module, classes, max_depth=2):
    if max_depth > 0:
        for name, obj in inspect.getmembers(module):
            if inspect.ismodule(obj):
                classes.union(classes, get_module_classes(obj, classes, max_depth-1))

            if inspect.isclass(obj) and issubclass(obj, NNInterface):
                classes.add(obj)
    return classes


def get_object(object_type, package, *args):
    classes = set()
    prefix = package.__name__ + "."
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        # print("Found submodule %s (is a package: %s)" % (modname, ispkg))
        module = __import__(modname)
        result = get_module_classes(module, set())
        classes = classes.union(result)

    for class_obj in classes:
        if object_type == class_obj.__name__:
            return class_obj(*args)