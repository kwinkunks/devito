
_types = [DimensionArgument]
class RuntimeArgProv(object):
    def prepare_runtime_dependencies(self, parameters):
        self.arg_s = runtime_argument(self, parameters)


def runtime_argument(obj, parameters):
    for typ in _types:
        if isinstance(obj, typ._classes):
            return typ(obj, parameters)
    return None

class RuntimeArgument(object):
    _classes = []
    def __init__(self, parent, parameters):
        self.parent = parent
        self._depends_on = tuple(self._dependencies())
        self.parameters = parameters
        self.values = {}
        self.refresh_deps()

    @property
    def depends_on(self):
        return self._depends_on
    
    def _dependencies(self):
        return []
    
    def arguments(self, rt_values):
        pass

    def ready(self):
        all([v is not None for v in self.values.values()])

class DimensionArgument(RuntimeArgument):
    _classes = [Dimension]

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
    
    def arguments(self, rt_values):
        if self.parent.size is not None:
            # If we have size, can't override at runtime
            assert(rt_values is None)
            # Make sure all symbolic data using this dimension are larger than our size
            assert(all([s > self.parent.size for s in self._dep_sym_data_sizes()]))
            self.values['size'] = size = self.parent.size
        else:
            size = rt_values or self.values['size']
            if size is None:
                other_dim = [d for d in self.depends_on if isinstance(d, Dimension)][0]
                if other_dim.arg_s.ready():
                    # If the other dimension knows its size, copy it from there
                    self.values['size'] = size = other_dim.arg_s.values['size']
                else:
                    # Use the minimum size of all the symbolic data using this dimension
                    self.values['size'] = size = min(self._dep_sym_data_sizes())
        return size

#TODO: Add similar class for SymbolicData and maybe CompositeData
        
    
