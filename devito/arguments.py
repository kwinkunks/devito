from logging import error
from devito.exceptions import InvalidArgument
from collections import OrderedDict


class RuntimeArgProv(object):
    def runtime_arg(self):
        return runtime_argument(self, self.name, self.dtype)


def runtime_argument(obj, name, dtype):
    for typ in _types:
        if isinstance(obj, typ._class):
            return typ(name, dtype)
    return None


class RuntimeArgument(object):
    def __init__(self, name, dtype):
        self._value = None
        self.name = name
        self.dtype = dtype
    
    @property
    def value(self):
        return self._value

    @property
    def ready(self):
        return self._value is not None

    @property
    def decl(self):
        raise NotImplemented()

    
class ScalarArgument(RuntimeArgument):
    @property
    def decl(self):
        return c.Value('const int', v.name)


class TensorArgument(RuntimeArgument):
    @property
    def decl(self):
        return c.Value(c.dtype_to_ctype(v.dtype), '*restrict %s_vec' % v.name)



class DimensionArgProvider(RuntimeArgProv):
    def _dependencies(self):
        deps = []
        
        if self.parent.size is not None:
            return deps
        # If we are a buffered dimension, we depend on the parent
        if isinstance(self.parent, BufferedDimension):
            deps.append(self.parent.parent)
        # Since we don't know our size, we depend on any SymbolicData that uses us
        for p in self.parameters:
            if isinstance(p, SymbolicData) and self.parent in p._indices:
                deps.append(p)
        return deps

    def _dep_sym_data_sizes(self):
        """ Get the shape parameter corresponding to the current dimension 
            for all dependent symbolic data
        """
        sd = [d for d in self.depends_on if isinstance(d, SymbolicData)]
        sizes = []
        for s in sd:
            index = find_index(self.parent, s._indices)
            sizes.append(s.shape[index])
        return sizes
    
    def process(self, arg, sizes):
        if self.size is not None:
            # If we have size, can't override at runtime
            assert(arg is None)
            # Make sure all symbolic data using this dimension are larger than our size
            assert(all([s > self.size for s in sizes]))
            size = self.size
        else:
            size = arg
            if size is None:
                other_dim = [d for d in self.depends_on if isinstance(d, Dimension)][0]
                if other_dim.arg_s.ready():
                    # If the other dimension knows its size, copy it from there
                    self.values['size'] = size = other_dim.arg_s.values['size']
                else:
                    # Use the minimum size of all the symbolic data using this dimension
                    self.values['size'] = size = min(self._dep_sym_data_sizes())
        return size

    
class SymbolicDataArgProvider(RuntimeArgProv):
    def process(self, arg):
        value = self
        extra_args = {}
        # Replace with provided argument if required and possible
        if arg is not None:
            if len(arg.shape) != len(self.shape):
                error('Rank of argument %s is %d, expected %d  '
                      % (self.name, len(arg.shape), len(self.shape)))
                raise InvalidArgument('Wrong data shape encountered')
            else:
                value = arg

                # If I have children, they should probably be replaced as well
                if self.is_CompositeData:
                    for c, nc in zip(self.children, value.children):
                        extra_args.update({c.name: nc})
        return value, extra_args
        
        


class RuntimeEngine(object):
    def __init__(self, parameters):
        self.symbolic_data = [x for x in parameters if isinstance(x, SymbolicDataArgProvider)]
        self.dimensions = [x for x in parameters if isinstance(x, DimensionArgProvider)]
        assert(all(isinstance(x, RuntimeArgProv)
                       for x in self.symbolic_data + self.dimensions))
        self.dimension_map = {}

        for s in self.symbolic_data:
            for d in s.indices:
                self.dimension_map.get(d, []).append(s)

    def _sym_data_sizes(self, dimension):
        sym_datas = self.dimension_map.get(dimension, [])
        return [x.shape[x.indices.index(dimension)] for x in sym_datas]
        

    def arguments(self, **kwargs):
        #s_args = [s.runtime_arg() for x in self.symbolic_data]
        #d_args = [d.runtime_arg() for x in self.dimensions]

        values = OrderedDict()
        extra_args = OrderedDict()
        for s in self.symbolic_data:
            # The value might be overriden either by a kwarg or by an arg
            # provided by another symbol
            overriden_value = kwargs.pop(s.name, None)

            if overriden_value is None:
                overriden_value = extra_args.pop(s.name, None)
                
            v, ea = s.process(overriden_value)
            values[s.name] = v
            
            # Does this symbol affect any other symbols?
            if len(ea) > 0:
                for k, v in ea.items():
                    if k in values.keys():
                        # We already visited the symbol for which this argument is.
                        # Just override
                        values[k] = v
                    else:
                        # We are yet to visit this symbol. Store it to use when we
                        # visit this symbol.
                        extra_args[k] = v

        # We should have used all the extra args by now
        assert(len(extra_args) == 0)

        overriden_value = None
        for d in self.dimensions:
            
            overriden_value = kwargs.pop(d.name, None)
            v_dict = d.process(overriden_value, self._sym_data_sizes(d))
            values.update(v_dict)
            

_types = [ScalarArgument, TensorArgument]
