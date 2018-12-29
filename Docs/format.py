import pytablewriter
import parse as pa
import extract as ext
import re
import inspect
import io


def format_class(cl):
    f = io.StringIO("")
    docs = cl.__doc__

    # skip undocumented classes
    if(docs == None):
        return f.getvalue()

    # # remove excess spaces
    docs = " ".join(docs.split())

    # General hig-level class docs
    f.write('# ' + cl.__name__ + '\n')
    f.write(docs + '\n')

    # Docs for __init__
    docs = (cl.__init__).__doc__
    clex = ext.PyBindExtract(docs)

    match = clex.extract("__init__")

    if(isinstance(match, list)):
        for ima, ma in enumerate(match):
            f.write(format_function(ma, 'Constructor [' + str(ima + 1) + ']'))
    else:
        f.write(format_function(match, 'Constructor'))

    # properties
    properties = inspect.getmembers(cl, lambda o: isinstance(o, property))
    f.write(format_properties(properties))
    return f.getvalue()


def format_function(ma, name):
    f = io.StringIO("")
    f.write('## ' + name)
    value_matrix = []
    signature = ma["signature"]

    sigp = pa.parse_signature(signature)
    gds = pa.GoogleDocString(ma["docstring"], args=sigp).parse()

    has_example = False
    for gd in gds:
        if(gd['header'] == 'Args'):

            for arg in gd['args']:
                field = arg['field']
                sig = arg['signature']
                # # remove excess spaces
                descr = " ".join(arg['description'].split())

                value_matrix.append([field, sig, descr])
        elif(gd['header'].startswith("Example")):
            examples = (gd['text'])
            has_example = True
        else:
            f.write(gd['text'] + '\n')

    writer = pytablewriter.MarkdownTableWriter()
    writer.header_list = ["Field", "Type", "Description"]
    writer.value_matrix = value_matrix
    writer.stream = f
    writer.write_table()
    if(has_example):
        f.write('### Examples' + '\n')
        f.write(examples + '\n')
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

    f.write('## Properties' + '\n')
    writer.write_table()
    return f.getvalue()
