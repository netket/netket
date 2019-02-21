import pytablewriter
import parse as pa
import extract as ext
import inspect
import io
import re


def format_class(cl):
    f = io.StringIO("")
    docs = cl.__doc__

    # skip undocumented classes
    if(docs == None):
        return f.getvalue()

    # remove excess spaces
    docs = re.sub(' +', ' ', docs).strip()

    # General high-level class docs
    f.write('# ' + cl.__name__ + '\n')
    f.write(docs + '\n\n')

    # Docs for __init__
    docs = (cl.__init__).__doc__

    if(not "__init__" in docs):
        return ""

    if(inspect.isabstract(cl)):
        return ""

    clex = ext.PyBindExtract(docs)

    match = clex.extract("__init__")

    if(isinstance(match, list)):
        for ima, ma in enumerate(match):
            f.write(format_function(
                ma, 'Class Constructor [' + str(ima + 1) + ']'))
    else:
        f.write(format_function(match, 'Class Constructor'))

    # methods
    f.write('## Class Methods \n')
    methods = inspect.getmembers(cl, predicate=inspect.isroutine)
    for method in methods:
        # skip special methods (__init__ is taken care of above)
        if(method[0].startswith('__')):
            continue

        docs = method[1].__doc__
        clex = ext.PyBindExtract(docs)

        match = clex.extract(method[0])
        f.write(format_function(match, method[0], level=3))

    # properties
    properties = inspect.getmembers(cl, lambda o: isinstance(o, property))
    f.write(format_properties(properties))
    return f.getvalue()


def format_function(ma, name, level=2):
    f = io.StringIO("")
    f.write('#' * level + ' ' + name)
    value_matrix = []

    gds = pa.GoogleDocString(ma["docstring"],
                             signature=ma['parsed_signature']).parse()

    has_example = False
    for gd in gds:
        if(gd['header'] == 'Args'):

            for arg in gd['args']:
                field = arg['field']
                sig = arg['signature']
                # remove excess spaces
                descr = " ".join(arg['description'].split())

                value_matrix.append([field, sig, descr])
        elif(gd['header'].startswith("Example")):
            examples = (gd['text'])
            has_example = True
        else:
            f.write(gd['text'] + '\n')

    writer = pytablewriter.MarkdownTableWriter()
    writer.header_list = ["Argument", "Type", "Description"]
    writer.value_matrix = value_matrix
    writer.stream = f
    if(len(value_matrix) > 0):
        writer.write_table()

    if(has_example):
        f.write('\n')
        f.write('### Examples' + '\n')
        f.write(examples + '\n')
    f.write('\n')
    return f.getvalue()


def format_properties(properties):
    writer = pytablewriter.MarkdownTableWriter()
    value_matrix = []

    for prop in properties:
        docs = prop[1].__doc__
        semic = docs.find(":")
        type_name = ''
        if(semic != -1):
            type_name = docs[: semic]
            docs = docs[semic + 1:]

        value_matrix.append([prop[0], type_name, docs])

    f = io.StringIO("")
    writer.header_list = ["Property", "Type", "Description"]
    writer.value_matrix = value_matrix
    writer.stream = f

    if(len(properties) > 0):
        f.write('## Properties' + '\n\n')
        writer.write_table()
    return f.getvalue()
