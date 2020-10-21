import os
import argparse
import yaml
import json
import ast
import textwrap
import inspect
from shlex import quote

from collections import OrderedDict

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
import json
from pprint import pprint
from sys import version_info
from inspect import ismethod


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _convert_value(value, strict=False):
    """Parse string as python literal if possible and fallback to string."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        if strict:
            raise
        # use as string if nothing else worked
        return value


class Config(MutableMapping, OrderedDict):
    @classmethod
    def load(cls, file_path):
        with open(file_path) as f:
            params = yaml.load(f.read(), Loader=yaml.FullLoader)

        # We expand ~ in those yaml entries with `path`
        # on their keys for making
        # config files more platform-independent
        params = {
            key: (
                os.path.expanduser(value)
                if "path" in key and value is not None
                else value
            )
            for key, value in params.items()
        }

        return cls(params)

    def dump(self, file_path):
        with open(file_path, "w") as f:
            d = self.to_dict()
            f.write(yaml.dump(d))

    def __init__(self, *args, **kwargs):
        self._map = OrderedDict()

        if args:
            d = args[0]
            # for recursive assignment handling
            trackedIDs = {id(d): self}
            if isinstance(d, dict):
                for k, v in self.__call_items(d):
                    if isinstance(v, dict):
                        if id(v) in trackedIDs:
                            v = trackedIDs[id(v)]
                        else:
                            v = self.__class__(v)
                            trackedIDs[id(v)] = v
                    if type(v) is list:
                        l = []
                        for i in v:
                            n = i
                            if isinstance(i, dict):
                                n = self.__class__(i)
                            l.append(n)
                        v = l
                    self._map[k] = v
        if kwargs:
            for k, v in self.__call_items(kwargs):
                self._map[k] = v

    _path_state = list()

    def __call_items(self, obj):
        if hasattr(obj, "iteritems") and ismethod(getattr(obj, "iteritems")):
            return obj.iteritems()
        else:
            return obj.items()

    def items(self):
        return self.iteritems()

    def iteritems(self):
        return self.__call_items(self._map)

    def __iter__(self):
        return self._map.__iter__()

    def next(self):
        return self._map.next()

    def __setitem__(self, k, v):
        # print('Called __setitem__')

        if (
            k in self._map
            and not self._map[k] is None
            and not isinstance(v, type(self._map[k]))
        ):
            if v is not None:
                raise ValueError(
                    f"Updating existing value {type(self._map[k])} "
                    f"with different type ({type(v)})."
                )
        split_path = k.split(".")
        current_option = self._map
        for p in split_path[:-1]:
            current_option = current_option[p]
        current_option[split_path[-1]] = v

    def __getitem__(self, k):
        split_path = k.split(".")
        current_option = self._map
        for p in split_path:
            if p not in current_option:
                raise KeyError(p)
            current_option = current_option[p]
        return current_option

    def __setattr__(self, k, v):
        if k in {"_map", "_ipython_canary_method_should_not_exist_"}:
            super(Config, self).__setattr__(k, v)
        else:
            self[k].update(v)

    def __getattr__(self, k):
        if k in {"_map", "_ipython_canary_method_should_not_exist_"}:
            return super(Config, self).__getattr__(k)

        try:
            v = super(self.__class__, self).__getattribute__(k)
            return v
        except AttributeError:
            self._path_state.append(k)
            pass

        return self[k]

    def __delattr__(self, key):
        return self._map.__delitem__(key)

    def __contains__(self, k):
        return self._map.__contains__(k)

    def __add__(self, other):
        if self.empty():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    def __str__(self):
        items = []
        for k, v in self.__call_items(self._map):
            # recursive assignment case
            if id(v) == id(self):
                items.append("{0}={1}(...)".format(k, self.__class__.__name__))
            else:
                items.append("{0}={1}".format(k, repr(v)))
        joined = ", ".join(items)
        out = "{0}({1})".format(self.__class__.__name__, joined)
        return out

    def __repr__(self):
        return str(self)

    def to_dict(self, flatten=False, parent_key="", sep="."):
        d = {}
        for k, v in self.items():
            if issubclass(type(v), Config):
                # bizarre recursive assignment support
                if id(v) == id(self):
                    v = d
                else:
                    v = v.to_dict()
            elif type(v) in (list, tuple):
                l = []
                for i in v:
                    n = i
                    if issubclass(type(i), Config):
                        n = i.to_dict()
                    l.append(n)
                if type(v) is tuple:
                    v = tuple(l)
                else:
                    v = l
            d[k] = v

        if flatten:
            d = flatten_dict(d, parent_key=parent_key, sep=sep)

        return d

    def pprint(self,):
        pprint(self.to_dict())

    def empty(self):
        return not any(self)

    # proper dict subclassing
    def values(self):
        return self._map.values()

    # ipython support
    def __dir__(self):
        return list(self.keys())

    def _ipython_key_completions_(self):
        return list(self.keys())

    @classmethod
    def parseOther(cls, other):
        if issubclass(type(other), Config):
            return other._map
        else:
            return other

    def __cmp__(self, other):
        other = Config.parseOther(other)
        return self._map.__cmp__(other)

    def __eq__(self, other):
        other = Config.parseOther(other)
        if not isinstance(other, dict):
            return False
        return self._map.__eq__(other)

    def __ge__(self, other):
        other = Config.parseOther(other)
        return self._map.__ge__(other)

    def __gt__(self, other):
        other = Config.parseOther(other)
        return self._map.__gt__(other)

    def __le__(self, other):
        other = Config.parseOther(other)
        return self._map.__le__(other)

    def __lt__(self, other):
        other = Config.parseOther(other)
        return self._map.__lt__(other)

    def __ne__(self, other):
        other = Config.parseOther(other)
        return self._map.__ne__(other)

    def __delitem__(self, key):
        return self._map.__delitem__(key)

    def __len__(self):
        return self._map.__len__()

    def clear(self):
        self._map.clear()

    def copy(self):
        return self.__class__(self)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

    def get(self, key, default=None):
        return self._map.get(key, default)

    def has_key(self, key):
        return key in self._map

    def iterkeys(self):
        return self._map.iterkeys()

    def itervalues(self):
        return self._map.itervalues()

    def keys(self):
        return self._map.keys()

    def pop(self, key, default=None):
        return self._map.pop(key, default)

    def popitem(self):
        return self._map.popitem()

    def setdefault(self, key, default=None):
        self._map.setdefault(key, default)

    def update(self, *args, **kwargs):
        if len(args) == 1:
            for key, value in args[0].items():
                if key in self and isinstance(self[key], dict):
                    if value is None:
                        self[key] = value
                    else:
                        self[key].update(value)
                else:
                    pass
                    raise ValueError()
        elif len(args) > 1:
            raise NotImplementedError
            # self._map.update(*args)
        else:
            raise NotImplementedError

    def viewitems(self):
        return self._map.viewitems()

    def viewkeys(self):
        return self._map.viewkeys()

    def viewvalues(self):
        return self._map.viewvalues()

    @classmethod
    def fromkeys(cls, seq, value=None):
        d = cls()
        d._map = OrderedDict.fromkeys(seq, value)
        return d

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    # bannerStr

    def _getListStr(self, items):
        out = "["
        mid = ""
        for i in items:
            mid += "  {}\n".format(i)
        if mid != "":
            mid = "\n" + mid
        out += mid
        out += "]"
        return out

    def _getValueStr(self, k, v):
        outV = v
        multiLine = len(str(v).split("\n")) > 1
        if multiLine:
            # push to next line
            outV = "\n" + v
        if type(v) is list:
            outV = self._getListStr(v)
        out = "{} {}".format(k, outV)
        return out

    def _getSubMapDotList(self, pre, name, subMap):
        outList = []
        if pre == "":
            pre = name
        else:
            pre = "{}.{}".format(pre, name)

        def stamp(pre, k, v):
            valStr = self._getValueStr(k, v)
            return "{}.{}".format(pre, valStr)

        for k, v in subMap.items():
            if isinstance(v, Config) and v != Config():
                subList = self._getSubMapDotList(pre, k, v)
                outList.extend(subList)
            else:
                outList.append(stamp(pre, k, v))
        return outList

    def _getSubMapStr(self, name, subMap):
        outList = ["== {} ==".format(name)]
        for k, v in subMap.items():
            if isinstance(v, self.__class__) and v != self.__class__():
                # break down to dots
                subList = self._getSubMapDotList("", k, v)
                # add the divit
                # subList = ['> {}'.format(i) for i in subList]
                outList.extend(subList)
            else:
                out = self._getValueStr(k, v)
                # out = '> {}'.format(out)
                out = "{}".format(out)
                outList.append(out)
        finalOut = "\n".join(outList)
        return finalOut

    def bannerStr(self):
        lines = []
        previous = None
        for k, v in self.items():
            if previous == self.__class__.__name__:
                lines.append("-")
            out = ""
            if isinstance(v, self.__class__):
                name = k
                subMap = v
                out = self._getSubMapStr(name, subMap)
                lines.append(out)
                previous = self.__class__.__name__
            else:
                out = self._getValueStr(k, v)
                lines.append(out)
                previous = "other"
        lines.append("--")
        s = "\n".join(lines)
        return s


class StrictConfig(Config):
    def update(self, *args, **kwargs):
        if len(args) == 1:
            for key, value in args[0].items():
                if self[key] and isinstance(value, type(self[key])):

                    if type(value) == dict or type(value) == StrictConfig:
                        self[key].update(value)
                    else:
                        self[key] = value
                elif type(self[key] == None):
                    self[key] = value
                elif not isinstance(value, type(self[key])):
                    raise ValueError(
                        f"Updating existing value {type(self[key])} "
                        f"with different type ({type(value)})."
                    )

        elif len(args) > 1:
            raise NotImplementedError
            # self._map.update(*args)
        else:
            raise NotImplementedError

    # def check_settings(self,*args,**kwargs):
    #     if self.settings:
    #         for setting in self.settings:
    #             if isinstance(setting, StrictConfig):
    #                 return

    # def check_setting(self,name,*args,**kwargs):
    #     if self.settings[name]:
    #         import ipdb; ipdb.set_trace()
    #         if isinstance(self[name],int):
    #             if 'max' in self.settings[name]:
    #                 if self[name] >= self.settings[name].max:
    #                     raise Exception(f'Value for {name} is above the max size: {self.settings[name].max}')
    #             if 'min' in self.settings[name]:
    #                 if self[name] <= self.settings[name].min:
    #                    raise Exception(f'Value for {name} is below the min size: {self.settings[name].min}')
    #         if 'choices' in self.settings[name]:
    #             if self[name] not in self.settings[name].choices:
    #                 raise Exception(f'Attribute {name} has to be in one of {self.settings[name].choices}')
    #     return True

    # def check_setting_to_set(self,name,value,*args,**kwargs):
    #     if name in self.settings:
    #         if isinstance(self[name],int):
    #             if 'max' in self.settings[name]:
    #                 if value >= self.settings[name].max:
    #                     raise Exception(f'Value for {name} is above the max size: {self.settings[name].max}')
    #             if 'min' in self.settings[name]:
    #                 if value <= self.settings[name].min:
    #                    raise Exception(f'Value for {name} is below the min size: {self.settings[name].min}')
    #         if 'choices' in self.settings[name]:
    #             if value not in self.settings[name].choices:
    #                 raise Exception(f'Attribute <{name}> has to be in one of {self.settings[name].choices}')


class ConfigParser(argparse.ArgumentParser):
    def __init__(self, default=None, *args, **kwargs):
        super(ConfigParser, self).__init__()

        if default is not None:
            self._default_config = Config.load(default)
        else:
            self._default_config = Config()

        self.add_argument(
            "--config",
            default=default,
            help="Read config from yaml file",
            metavar="",
        )

        self.add_argument(
            "--set",
            dest="config_updates",
            nargs="+",
            action=StoreDictKeyPair,
            metavar="KEY=VAL",
        )

    def parse_args(self):
        args = super(ConfigParser, self).parse_args()

        if args.config is not None:
            config = Config.load(args.config)
            self._default_config.update(config)

        if args.config_updates is not None:
            for key, value in args.config_updates.items():
                self._default_config[key] = value

        return self._default_config


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        config_updates = {}
        for kv in values:
            path, value = kv.split("=")
            if type(value) == dict:
                path = path.strip()
                config_updates[path] = value
            else:
                path = path.strip()  # get rid of surrounding whitespace
                value = value.strip()  # get rid of surrounding whitespace
                config_updates[path] = _convert_value(value)

        setattr(namespace, self.dest, config_updates)


class StrictConfigParser(argparse.ArgumentParser):
    def __init__(self, default=None, *args, **kwargs):
        super(StrictConfigParser, self).__init__()

        if default is not None:
            self._default_config = StrictConfig.load(default)
        else:
            raise RuntimeError("There is no class specified")

        self.add_argument(
            "--config",
            default=default,
            help="Read config from yaml file",
            metavar="",
        )

        self.add_argument(
            "--set",
            dest="config_updates",
            nargs="+",
            action=StoreDictKeyPair,
            metavar="KEY=VAL",
        )

    def parse_args(self):
        args = super(StrictConfigParser, self).parse_args()
        if args.config is not None and args.config_updates is not None:
            self._default_config = StrictConfig.load(args.config)
            for key, value in args.config_updates.items():
                to_update = StrictConfig({key: value})
                self._default_config.update(to_update)

        elif args.config is not None:
            config = StrictConfig.load(args.config)
            self._default_config = config

        return self._default_config
