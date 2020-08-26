#!/bin/sh -e
# Change the number of parallel run

txt2html() {
    # Insert the page path into the source URL.
    sed -E "s	%%TITLE%%	$title	"
}


