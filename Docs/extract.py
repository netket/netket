"""
MIT License

Copyright (c) 2018 Ossian O'Reilly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
This module is used to extract a docstring from source.
"""
import re


class Extract(object):
    """
    Base class for extracting docstrings.

    Attributes:
        txt : A string that contains the source code that has been read from
            `source`.
        query : A copy of the most recent docstring searched. The search is
            specified in the form of `Class.method`, or `function`, or `.` to
            search for the module docstring.
        classname : Holds the class name of the query.
        funcname : Holds the function or method name of the query.
        dtype : Holds the type of the query `module`, `class`, `method`, or
            `function`.

    """

    def __init__(self, txt):
        """
        Initializer for Extract.

        Arguments:
            txt: A string containing the text to extract docstrings from.

        """
        self.txt = txt
        self.query = ''
        self.classname = ''
        self.funcname = ''
        self.dtype = ''
        #This dictionary contains the ids of the different attributes that can
        #be captured. The value specifies the order in which attribute is
        #captured.
        self.ids = {
            'class': 0,
            'function': 1,
            'signature': 2,
            'indent': 3,
            'docstring': 4,
            'body': 5,
            'return_annotation': 100
        }
        # This dictionary contains keywords that describe if a function should
        # start with 'def', and if the docstring that comes after the function
        # should be enclosed with """.
        self.keywords = {
            'function': 'def ',
            'docstring': '"""',
            'signature_end': ':',
            'token_split': 'def '
        }
        self.function_keyword = 'def '
        self.docstring_keyword = '"""'
        # Handle multiple functions (typically function overloads) by splitting
        # text for each function. The key `token_split` in `self.keywords`.
        # determines the regex for the splitting.
        self.split = 1

    def get_matches(self, pattern):
        """
        Apply regex pattern for finding functions, classes, etc. 

        Args:
            pattern: The pattern to search for.

        Raises:
            NameError: if no matches are found. 

        Returns:
            The matches found.
            

        """
        matches = re.compile(pattern, re.M).findall(self.txt)
        if not matches:
            raise NameError(
                r'Unable to extract docstring for `%s`' % self.query)
        return matches

    def extract(self, query):
        """
        Extracts the docstring.

        Arguments:
            query : The docstring to search for. The search is specified in the
                form of `Class.method`, or `function`, or `.` to search for the
                module docstring.

        Returns:
            A dictionary that matches the description given by `Extract.find`.

        """

        self.query = query
        self.classname, self.funcname, self.dtype = get_names(query)
        types = {
            'class': self.extract_class,
            'method': self.extract_method,
            'function': self.extract_function,
            'module': self.extract_module
        }

        return types[self.dtype]()

    def extract_function(self):
        """
        Override this method to extract function docstrings for the specific
        language. The functions extracted are module functions. Lamba functions
        are not extracted.

        Returns:
            A dictionary that matches the description given by `Extract.find`.
        """
        pass

    def extract_class(self):
        """
        Override this method to extract class docstrings for the specific
        language.

        Returns:
            A dictionary that matches the description given by `Extract.find`.
        """
        pass

    def extract_method(self):
        """
        Override this method to extract method docstrings for the specific
        language.

        Returns:
            A dictionary that matches the description given by `Extract.find`.
        """
        pass

    def extract_module(self):
        """
        Override this method to extract module docstrings for the specific
        language. Module docstrings are defined at the start of a file and are
        not attached to any block of code.

        Returns:
            A dictionary that matches the description given by `Extract.find`.
        """
        pass

    def findall(self, pattern, ids=None):
        """
        Splits the input text into multiple strings and performs a search for
        each string. The idea is to handle text that contains multiple
        functions/methods and search each such function for a specific pattern.

        """
        import re

        out = []
        input_txt = self.txt
        if self.split:
            matches = re.split(
                self.keywords['token_split'], self.txt, flags=re.M)
            for match in matches:
                self.txt = match
                try:
                    out.append(self.find(pattern, ids))
                except:
                    continue
            self.txt = input_txt
            if not out:
                raise NameError(
                    r'Unable to extract docstring for `%s`' % self.query)
            if len(out) == 1:
                return out[0]
            else:
                return out
        else:
            return self.find(pattern, ids)

    def find(self, pattern, ids=None):
        """
        Performs a search for a docstring that matches a specific pattern.

        Returns:
            dict: The return type is a dictionary with the following keys:
                 * `class` :  The name of the class.
                 * `function` : The name of the function/method.
                 * `signature` : The signature of the function/method.
                 * `docstring` : The docstring itself.
                 * `type` : What type of construct the docstring is attached to.
                      This can be either `'module'`, `'class'`, `'method'`, or
                      `'function'`.
                 * `label` : The search query string.
                 * `source` : The source code if the query is a function/method.
                 * `args` : A dictionary containing signature arguments, and
                    return type.

        Raises:
            NameError: This is exception is raised if the docstring cannot be
                extracted.
        """
        import textwrap
        try:
            from . import parse
        except:
            import parse
        matches = self.get_matches(pattern)

        if not ids:
            ids = self.ids

        out_list = []

        for match in matches:
            cls = get_match(match, ids['class'])
            function = get_match(match, ids['function'])
            signature = format_txt(get_match(match, ids['signature']))
            indent = len(get_match(match, ids['indent']))
            return_annotation = get_match(match, ids['return_annotation'])
            docstring = remove_indent(
                get_match(match, ids['docstring']), indent)
            if self.dtype == 'function' or self.dtype == 'method':
                source = textwrap.dedent(self.function_keyword + function +
                                         signature + ':' + return_annotation +
                                         '\n' + get_match(match, ids['body']))
            else:
                source = ''

            out = {}
            out['class'] = cls
            out['function'] = function
            out['signature'] = signature
            out['docstring'] = docstring
            out['return_annotation'] = return_annotation
            out['source'] = source
            out['type'] = self.dtype
            out['label'] = self.query
            try:
                out['parsed_signature'] = parse.parse_signature(
                    out['signature'])
            except:
                pass
            out_list.append(out)

        if len(out_list) == 1:
            return out_list[0]
        else:
            return out_list


class PyExtract(Extract):
    """
    Base class for extracting docstrings from python source code.
    """

    def extract_function(self):
        pattern = (r'^\s*(%s)(\([\w\W]*?\)' % self.funcname +
                   r'\s*(?:->\s*(\w+))?)%s\n+' % self.keywords['signature_end']
                   + r'(\s+)%s([\w\W]*)?%s\n((\4.*\n+)+)?' %
                   (self.keywords['docstring'], self.keywords['docstring']))

        ids = {
            'class': 100,
            'function': 0,
            'signature': 1,
            'return_annotation': 2,
            'indent': 3,
            'docstring': 4,
            'body': 5
        }
        return self.findall(pattern, ids)

    def extract_class(self):
        pattern = (r'^\s*class\s+(%s)()(\(\w*\))?:\n(\s+)"""([\w\W]*?)"""()' %
                   self.classname)
        return self.find(pattern)

    def extract_method(self):
        pattern = (r'class\s+(%s)\(?\w*\)?:[\n\s]+[\w\W]*?' % self.classname +
                   r'[\n\s]+def\s+(%s)(\(self[\w\W]*?\)' % self.funcname +
                   r'\s*(?:->\s*(\w+))?)%s\n+' % self.keywords['signature_end']
                   + r'(\s+)"""([\w\W]*?)"""\n((?:\4.*\n+)+)?')
        ids = {
            'class': 0,
            'function': 1,
            'signature': 2,
            'return_annotation': 3,
            'indent': 4,
            'docstring': 5,
            'body': 6
        }
        return self.find(pattern, ids)

    def extract_module(self):
        pattern = r'()()()()^"""([\w\W]*?)"""'
        return self.find(pattern)


class PyBindExtract(PyExtract):
    """
    Extract function header, signature, and documentation from PyBind-generated
    docstrings.
    """

    def __init__(self, query):
        PyExtract.__init__(self, query)
        self.function_keyword = ""
        self.docstring_symbol = ""
        self.keywords = {
            'function': '',
            'docstring': '',
            'self': 'self',
            'signature_end': '',
            'token_split': '^\s*\d+\.'
        }
        self.split = 1

    def extract_function(self):
        pattern = (
            r'^\s*(%s)(\([\w\W]*?\)' % (self.funcname) +
            r'\s*(?:->\s*([\w\W]*?)))%s\n+' % self.keywords['signature_end'] +
            r'(\s*)%s([\w\W]*)?%s((\4.*\n+)+)?' %
            (self.keywords['docstring'], self.keywords['docstring']))
        ids = {
            'class': 100,
            'function': 0,
            'signature': 1,
            'return_annotation': 2,
            'indent': 3,
            'docstring': 4,
            'body': 5
        }
        return self.findall(pattern, ids)

    def extract_method(self):
        out = self.extract_function()
        pattern = r'self:.*(%s)\s*,+' % (self.classname)
        match = re.compile(pattern, re.M).findall(out['signature'])
        if not match:
            raise NameError('Class name in query string does not match class ' \
                            'name in docstring.')
        out['class'] = self.classname
        return out

    def extract_class(self):
        pattern = (r'^\s*class\s+(%s)' % self.classname + r'(\(\w+\))?\n+' +
                   r'(\s+)([\w\W]+)*')
        ids = {
            'class': 0,
            'function': 100,
            'signature': 1,
            'return_annotation': 100,
            'indent': 2,
            'docstring': 3,
            'body': 4
        }
        return self.find(pattern, ids)

    def extract_overloaded_function(self):
        pattern = r'\s*Overloaded function.\n+\s*((\d+\.)[\w\W]+)'
        matches = self.get_matches(pattern)
        functions = matches[0][0]
        lines = self.txt.split('\n')
        pattern = r'\s*\d+\.\s+(%s\(.*\)\s*(?:->\s+\w+))' % (self.funcname)
        search = re.compile(pattern, re.M)
        first = 0
        function = []
        begin = 1

        functions = []
        # Look for function header: 1. function_name(..) -> ..
        # capture text following it, and then parse each one individually
        for line in lines:
            matches = search.findall(line)

            if matches:
                function_header = matches[0]
                # Done parsing function
                if not begin:
                    self.txt = '\n'.join(function)
                    functions.append(self.extract_function())
                # Start parsing next function
                function = []
                function.append(function_header)
                if begin:
                    begin = 0
            elif not begin:
                function.append(line)

        self.txt = '\n'.join(function)
        functions.append(self.extract_function())
        return functions


def extract(filestr, query):
    """
    Extracts a docstring from source.

    Arguments:
        filestr: A string that specifies filename of the source code to extract
            from.
        query: A string that specifies what type of docstring to extract.

    """
    import os

    filename = os.path.splitext(filestr)
    ext = filename[1]

    options = {'.py': PyExtract}

    if ext in options:
        extractor = options[ext](open(filestr).read())

    return extractor.extract(query)


def get_names(query):
    """
    Extracts the function and class name from a query string.
    The query string is in the format `Class.function`.
    Functions starts with a lower case letter and classes starts
    with an upper case letter.

    Arguments:
        query: The string to process.

    Returns:
        tuple: A tuple containing the class name, function name,
               and type. The class name or function name can be empty.

    """
    funcname = ''
    classname = ''
    dtype = ''

    members = query.split('.')
    if len(members) == 1:
        # If no class, or function is specified, then it is a module docstring
        if members[0] == '':
            dtype = 'module'
        # Identify class by checking if first letter is upper case
        elif members[0][0].isupper():
            classname = query
            dtype = 'class'
        else:
            funcname = query
            dtype = 'function'
    elif len(members) == 2:
        # Parse method
        classname = members[0]
        funcname = members[1]
        dtype = 'method'
    else:
        raise ValueError('Unable to parse: `%s`' % query)

    return (classname, funcname, dtype)


def remove_indent(txt, indent):
    """
    Dedents a string by a certain amount.
    """
    lines = txt.split('\n')
    if lines[0] != '\n':
        header = '\n' + lines[0]
    else:
        header = ''
    return '\n'.join([header] + [line[indent:] for line in lines[1:]])


def get_match(match, index, default=''):
    """
    Returns a value from match list for a given index. In the list is out of
    bounds `default` is returned.

    """
    if index >= len(match):
        return default
    else:
        return match[index]


def format_txt(signature):
    """
    Remove excess spaces and newline characters.
    """
    return ' '.join(' '.join(signature.split('\n')).split())
