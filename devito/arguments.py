from logging import error
from devito.exceptions import InvalidArgument
from collections import OrderedDict
from cached_property import cached_property
from tools import flatten
import sys


class RuntimeArgProv(object):
    
    @property
    def rtargs(self):
        raise NotImplemented()


class RuntimeArgument(object):
    
    def __init__(self, name, source, default_value = None):
        self.name = name
        self.source = source
        self._value = self._default_value = default_value
        
    @property
    def value(self):
        return self._value

    @property
    def ready(self):
        return self._value is not None

    @property
    def decl(self):
        raise NotImplemented()

    def reset(self):
        self._value = self._default_value

    def verify(self, kwargs):
        raise NotImplemented()
        

class ScalarArgument(RuntimeArgument):
    
    def __init__(self, name, source, default_value, reducer):
        super(ScalarArgument, self).__init__(name, source, default_value)
        self.reducer = reducer
        
    @property
    def decl(self):
        return c.Value('const int', v.name)

    def verify(self, value):
        # Assuming self._value was initialised as appropriate for the reducer
        if value is not None:
            self._value = self.reducer(self._value, value)

        return self._value is not None


class TensorArgument(RuntimeArgument):
    
    def __init__(self, name, source, dtype):
        super(TensorArgument, self).__init__(name, source)
        self.dtype = dtype
        self._value = self.source

    @property
    def value(self):
        return self._value.data

    @property
    def decl(self):
        return c.Value(c.dtype_to_ctype(v.dtype), '*restrict %s_vec' % v.name)

    def verify(self, value):
        if value is None:
            # Assuming self._value is initialized to self.source
            value = self._value

        verify = self.source.shape == value.shape
        # Side-effect alert: We are modifying kwargs to read the value of children
        # Can we do without this? 
        if self.source.is_CompositeData:
            for child, orig_child in zip(value.children, self.source.children):
                    orig_child.rtargs[0].verify(child)
                    
        verify = verify and all([d.verify(v) for d, v in zip(self.source.indices, value.shape)])     
        if verify:    
            self._value = value

        return self._value is not None and verify


class DimensionArgProvider(RuntimeArgProv):
    reducer = min

    def __init__(self, *args, **kwargs):
        super(DimensionArgProvider, self).__init__(*args, **kwargs)
        self._value = sys.maxint

    @property
    def value(self):
        return self._value
    
    @cached_property
    def rtargs(self):
    # TODO: Create proper Argument objects - with good init values
        if self.size is not None:
            return []
        else:
            start = ScalarArgument("%s_s" % self.name, self, 0, max)
            end = ScalarArgument("%s_e" % self.name, self, sys.maxint, min)
            return [start, end]

    # TODO: Do I need a verify on a dimension?
    def verify(self, value):
        verify = True

        if value is None and self._value is not None:
            return verify

        if value is not None and value == self._value:
            return verify
        
        if self.size is not None:
        # Assuming the only people calling my verify are symbolic data, they need to be bigger than my size if I have a hard-coded size
            verify = (value > self.size)
        else:
            # Assuming self._value was initialised as maxint
            value = self.reducer(self._value, value)
            if hasattr(self, 'parent'):
                verify = verify and self.parent.verify(value)
                
                # If I don't know my value, ask my parent
                if value is None:
                    value = self.parent.value
                
            # Derived dimensions could be linked through constraints
            # At this point, a constraint needs to be added that limits dim_e - dim_s < SOME_MAX
            # Also need a default constraint that dim_e > dim_s (or vice-versa)
            
            verify = verify and all([a.verify(v) for a, v in zip(self.rtargs, (0, value))])
            if verify:
                self._value = value
            assert(verify)
        return verify               

    
class SymbolicDataArgProvider(RuntimeArgProv):
    
    @cached_property
    def rtargs(self):
    # TODO: Create proper Argument objects - with good init values
        return [TensorArgument(self.name, self, self.dtype)]

    
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

        self.tensor_arguments = flatten([x.rtargs for x in self.symbolic_data])
        self.scalar_arguments = flatten([x.rtargs for x in self.dimensions])
        
    def arguments(self, *args, **kwargs):
        for ta in self.tensor_arguments:
            assert(ta.verify(kwargs.pop(ta.name, None)))

        for d in self.dimensions:
            assert(d.verify(kwargs.pop(d.name, None)))

        for s in self.scalar_arguments:
            assert(s.verify(kwargs.pop(s.name, None)))

        return OrderedDict([(x.name, x.value) for x in self.tensor_arguments + self.scalar_arguments])
            

_types = [ScalarArgument, TensorArgument]
