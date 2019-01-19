"""
usage: doc_diff.py [module1 module2 ...]

Check documentation for changes.

module1, ...   List the submodules to check for changes. Checks documentation
               for all submodules if left empty.

Use this tool to make sure that any updates to the documentation
generating tools do not cause any unwanted modifications. 

How to use it:

1. Re-generate the documentation to make sure that it is up to date.
2. Apply the update patch to the documentation tools.
3. Run this tool.

Example output:
 python3 doc_diff.py graph
 Docs for  <class 'netket.graph.Graph'>
 Docs for  <class 'netket.graph.Hypercube'>
 Mismatch found in: netket.graph.Hypercube
 Report written to: report/graph/Hypercube.html
 Building documentation for: netket.graph.CustomGraph
 Mismatch found in: netket.graph.CustomGraph
 Report written to: report/graph/CustomGraph.html

"""

import filecmp
import difflib
import os
import sys
import format as fmt
import netket
import shutil
import build_docs

build_dir = 'temp'
report_dir = 'report'


def get_generated_docs(submodules):
    """
    Return a list of paths to generated docs (e.g., `Graph/Hypercube.md`), paths
    to docs to generate, and module import statement (e.g., `graph.hypercube`). 

    """
    # Reference files
    ref_files = []
    # Potentially modified files (these are temporarily written to disk)
    mod_files = []
    # Module import statements
    classes = []
    for submodule in submodules:
        tmp = os.listdir(submodule)
        for file_ in tmp:
            ref_files.append('%s/%s'%(submodule, file_))
            mod_files.append('%s/%s/%s'%(build_dir, submodule, file_))
            classes.append('netket.%s.%s'%(submodule.lower(),
                           file_.split('.')[0])) 

    return ref_files, mod_files, classes

def init_dir(new_dir):
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

def make_report(ref_file, mod_file, class_name, report_dir='reports/',
                verbose=1):
    """
    Compare previously generated docs with most recent version. Generate a HTML
    report for any files that do not match. 

    Args:
        ref_file: path to reference file.
        mod_file: path to modified file.

    Returns:
        A flag that is set to `True` if any inconsistencies are encountered.

    """
    is_consistent = filecmp.cmp(ref_file, mod_file)

    if not is_consistent:
        init_dir(report_dir)
        fromlines = open(ref_file).read().split('\n')
        tolines = open(mod_file).read().split('\n')
        report = difflib.HtmlDiff()
        html = report.make_file(fromlines, tolines)
        # remove netket from path
        out_file = '%s/%s'%(report_dir, 
                           '/'.join(class_name.replace('.',
                           '/').split('/')[1::]))
        out_dir = os.path.dirname(out_file)

        print("Mismatch found in: %s"%class_name)
        init_dir(out_dir)
        with open(out_file + '.html', 'w') as filehandle:
            filehandle.write(html)

        if verbose:
            print("Report written to: %s.html"%out_file)

    return is_consistent

def init_docs(build_dir, doc_dirs):
    init_dir(build_dir)
    for doc_dir in doc_dirs:
        init_dir('%s/%s'%(build_dir, doc_dir))

def run(build_dir, submodules, report_dir):
    init_docs(build_dir, submodules)
    build_docs.build_docs(output_directory=build_dir, submodules=submodules)
    ref_files, mod_files, classes = get_generated_docs(submodules)
    err = False
    for ref_file, mod_file, class_name in zip(ref_files, mod_files, classes):
        is_consistent = make_report(ref_file, mod_file, class_name, report_dir)
        if not is_consistent:
            err = True

    # Remove directory that contains temporarily generated docs. 
    shutil.rmtree(build_dir)
    return err

if len(sys.argv) > 1:
    submodules = sys.argv[1::]
else:
    submodules = build_docs.default_submodules

sys.exit(run(build_dir, submodules, report_dir))
