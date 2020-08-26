#!/bin/sh -e
#
# Change the number of parallel run

num_config=`ls config/triton | wc -l`

mod_job() {
    # Insert the page path into the source URL.
    sed -E "s	%%NUM%%	$num_config	"
}

txt2html < "site/$page" > "docs/${page%%.txt}.html"


