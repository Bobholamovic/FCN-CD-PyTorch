import logging
import os
from time import localtime
from collections import OrderedDict
from weakref import proxy

FORMAT_LONG = "[%(asctime)-15s %(funcName)s] %(message)s"
FORMAT_SHORT = "%(message)s"


class Logger:
    _count = 0

    def __init__(self, scrn=True, log_dir='', phase=''):
        super().__init__()
        self._logger = logging.getLogger('logger_{}'.format(Logger._count))
        Logger._count += 1
        self._logger.setLevel(logging.DEBUG)

        if scrn:
            self._scrn_handler = logging.StreamHandler()
            self._scrn_handler.setLevel(logging.INFO)
            self._scrn_handler.setFormatter(logging.Formatter(fmt=FORMAT_SHORT))
            self._logger.addHandler(self._scrn_handler)
            
        if log_dir and phase:
            self.log_path = os.path.join(log_dir,
                    '{}-{:-4d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}.log'.format(
                        phase, *localtime()[:6]
                      ))
            self.show_nl("log into {}\n\n".format(self.log_path))
            self._file_handler = logging.FileHandler(filename=self.log_path)
            self._file_handler.setLevel(logging.DEBUG)
            self._file_handler.setFormatter(logging.Formatter(fmt=FORMAT_LONG))
            self._logger.addHandler(self._file_handler)

    def show(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)

    def show_nl(self, *args, **kwargs):
        self._logger.info("")
        return self.show(*args, **kwargs)

    def dump(self, *args, **kwargs):
        return self._logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self._logger.error(*args, **kwargs)

    @staticmethod
    def make_desc(counter, total, *triples):
        desc = "[{}/{}]".format(counter, total)
        # The three elements of each triple are
        # (name to display, AverageMeter object, formatting string)
        for name, obj, fmt in triples:
            desc += (" {} {obj.val:"+fmt+"} ({obj.avg:"+fmt+"})").format(name, obj=obj)
        return desc

_default_logger = Logger()


class _WeakAttribute:
    def __get__(self, instance, owner):
        return instance.__dict__[self.name]
    def __set__(self, instance, value):
        if value is not None:
            value = proxy(value)
        instance.__dict__[self.name] = value
    def __set_name__(self, owner, name):
        self.name = name


class _TreeNode:
    _sep = '/'
    _none = None

    parent = _WeakAttribute()   # To avoid circular reference

    def __init__(self, name, value=None, parent=None, children=None):
        super().__init__()
        self.name = name
        self.val = value
        self.parent = parent
        self.children = children if isinstance(children, dict) else {}
        if isinstance(children, list):
            for child in children:
                self._add_child(child)
        self.path = name
    
    def get_child(self, name, def_val=None):
        return self.children.get(name, def_val)

    def set_child(self, name, val=None):
        r"""
            Set the value of an existing node. 
            If the node does not exist, return nothing
        """
        child = self.get_child(name)
        if child is not None:
            child.val = val

        return child

    def add_place_holder(self, name):
        return self.add_child(name, val=self._none)

    def add_child(self, name, val):
        r"""
            If not exists or is a placeholder, create it
            Otherwise skips and returns the existing node
        """
        child = self.get_child(name, None)
        if child is None:
            child = _TreeNode(name, val, parent=self)
            self._add_child(child)
        elif child.val == self._none:
            # Retain the links of the placeholder
            # i.e. just fill in it
            child.val = val

        return child

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        try:
            repr = self.path + ' ' + str(self.val)
        except TypeError:
            repr = self.path
        return repr

    def __contains__(self, name):
        return name in self.children.keys()

    def __getitem__(self, key):
        return self.get_child(key)

    def _add_child(self, node):
        r""" Into children dictionary and set path and parent """
        self.children.update({
            node.name: node
        })
        node.path = self._sep.join([self.path, node.name])
        node.parent = self

    def apply(self, func):
        r"""
            Apply a callback function on ALL descendants
            This is useful for the recursive traversal
        """
        ret = [func(self)]
        for _, node in self.children.items():
            ret.extend(node.apply(func))
        return ret

    def bfs_tracker(self):
        queue = []
        queue.insert(0, self)
        while(queue):
            curr = queue.pop()
            yield curr
            if curr.is_leaf():
                continue
            for c in curr.children.values():
                queue.insert(0, c)


class _Tree:
    def __init__(
        self, name, value=None, strc_ele=None, 
        sep=_TreeNode._sep, def_val=_TreeNode._none
    ):
        super().__init__()
        self._sep = sep
        self._def_val = def_val
        
        self.root = _TreeNode(name, value, parent=None, children={})
        if strc_ele is not None:
            assert isinstance(strc_ele, dict)
            # This is to avoid mutable parameter default
            self.build_tree(OrderedDict(strc_ele or {}))

    def build_tree(self, elements):
        # The siblings could be out-of-order
        for path, ele in elements.items():
            self.add_node(path, ele)

    def get_root(self):
        r""" Get separated root node """
        return _TreeNode(
            self.root.name, self.root.value, 
            parent=None, children=None
        )

    def __repr__(self):
        return self.__dumps__()
        
    def __dumps__(self):
        r""" Dump to string """
        _str = ''
        # DFS
        stack = []
        stack.append((self.root, 0))
        while(stack):
            root, layer = stack.pop()
            _str += ' '*layer + '-' + root.__repr__() + '\n'

            if root.is_leaf():
                continue
            # Note that the order of the siblings is not retained
            for c in reversed(list(root.children.values())):
                stack.append((c, layer+1))

        return _str

    def vis(self):
        r""" Visualize the structure of the tree """
        _default_logger.show(self.__dumps__())

    def __contains__(self, obj):
        return any(self.perform(lambda node: obj in node))

    def perform(self, func):
        return self.root.apply(func)

    def get_node(self, tar, mode='name'):
        r"""
            This is different from the travasal in that
            the search allows early stop
        """
        if mode == 'path':
            nodes = self.parse_path(tar)
            root = self.root
            for r in nodes:
                if root is None:
                    root = root.get_child(r)
            return root
        else:
            # BFS
            bfs_tracker = self.root.bfs_tracker()
            # bfs_tracker.send(None)

            for node in bfs_tracker:
                if getattr(node, mode) == tar:
                    return node
        return

    def set_node(self, path, val):
        node = self.get_node(path, mode=path)
        if node is not None:
            node.val = val
        return node

    def add_node(self, path, val=None):
        if not path.strip():
            raise ValueError("the path is null")
        if val is None:
            val = self._def_val
        names = self.parse_path(path)
        root = self.root
        nodes = [root]
        for name in names[:-1]:
            # Add placeholders
            root = root.add_child(name, self._def_val)
            nodes.append(root)
        root = root.add_child(names[-1], val)
        return root, nodes

    def parse_path(self, path):
        return path.split(self._sep)

    def join(self, *args):
        return self._sep.join(args)
        
        
class OutPathGetter:
    def __init__(self, root='', log='logs', out='outs', weight='weights', suffix='', **subs):
        super().__init__()
        self._root = root.rstrip('/')    # Work robustly for multiple ending '/'s
        self._suffix = suffix
        self._keys = dict(log=log, out=out, weight=weight, **subs)
        self._dir_tree = _Tree(
            self._root, 'root',
            strc_ele=dict(zip(self._keys.values(), self._keys.keys())),
            sep='/', 
            def_val=''
        )

        self.update_keys(False)
        self.update_tree(False)

        self.__counter = 0

    def __str__(self):
        return '\n'+self.sub_dirs

    @property
    def sub_dirs(self):
        return str(self._dir_tree)

    @property
    def root(self):
        return self._root

    def _update_key(self, key, val, add=False, prefix=False):
        if prefix:
            val = os.path.join(self._root, val)
        if add:
            # Do not edit if exists
            self._keys.setdefault(key, val)
        else:
            self._keys.__setitem__(key, val)

    def _add_node(self, key, val, prefix=False):
        if not prefix and key.startswith(self._root):
            key = key[len(self._root)+1:]
        return self._dir_tree.add_node(key, val)

    def update_keys(self, verbose=False):
        for k, v in self._keys.items():
            self._update_key(k, v, prefix=True)
        if verbose:
            _default_logger.show(self._keys)
        
    def update_tree(self, verbose=False):
        self._dir_tree.perform(lambda x: self.make_dir(x.path))
        if verbose:
            _default_logger.show("\nFolder structure:")
            _default_logger.show(self._dir_tree)

    @staticmethod
    def make_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def get_dir(self, key):
        return self._keys.get(key, '') if key != 'root' else self.root

    def get_path(
        self, key, file, 
        name='', auto_make=False, 
        suffix=True, underline=False
    ):
        folder = self.get_dir(key)
        if len(folder) < 1:
            raise KeyError("key not found") 
        if suffix:
            path = os.path.join(folder, self.add_suffix(file, underline=underline))
        else:
            path = os.path.join(folder, file)

        if auto_make:
            base_dir = os.path.dirname(path)

            if base_dir in self:
                return path
            if name:
                self._update_key(name, base_dir, add=True)
            '''
            else:
                name = 'new_{:03d}'.format(self.__counter)
                self._update_key(name, base_dir, add=True)
                self.__counter += 1
            '''
            des, visit = self._add_node(base_dir, name)
            # Create directories along the visiting path
            for d in visit: self.make_dir(d.path)
            self.make_dir(des.path)
        return path

    def add_suffix(self, path, suffix='', underline=False):
        pos = path.rfind('.')
        if pos == -1:
            pos = len(path)
        _suffix = self._suffix if len(suffix) < 1 else suffix
        return path[:pos] + ('_' if underline and _suffix else '') + _suffix + path[pos:]

    def __contains__(self, value):
        return value in self._keys.values()


class Registry(dict):
    def register(self, key, val):
        if key in self: _default_logger.warning("key {} already registered".format(key))
        self[key] = val


R = Registry()
R.register('DEFAULT_LOGGER', _default_logger)
register = R.register