# Read MXNet Code
- All of the module interfaces are listed in [include](../../include). The
  interfaces are heavily documented.
- Read the
  [Doxygen Version](https://mxnet.readthedocs.org/en/latest/doxygen) of the
  document.
- Each module depends on other modules as defined in the header files in
  [include](../../include).
- Module implementation is in the  [src](../../src) folder.
- Source code sees only the file within its folder,
  [src/common](../../src/common) and [include](../../include).

Most modules are mostly self-contained, with interface dependency on the engine. 
You're free to pick the one you are interested in, and read that part.

## Other Resources
* [Doxygen Version of C++ API](https://mxnet.readthedocs.org/en/latest/doxygen) comprehensively documents the C++ API.

## Next Steps

* [Develop and Hack MXNet](http://mxnet.io/how_to/develop_and_hack.html)