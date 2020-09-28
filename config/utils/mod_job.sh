#!/bin/sh -e
#
# Change the number of parallel run

num_config=`ls $2 | wc -l`
num_array=`echo $num_config - 1 | bc`

[ -z ${1+x} ] && echo "Error: Missing first argument (template file path)" && exit
[ -z ${2+x} ] && echo "Error: Missing second argument (config dir path)" && exit
[ -z ${3+x} ] && echo "Error: Missing third argument (output file path)" && exit

mod_job() {
    # Insert the array size in template
    sed -E "s	%%NUM_ARRAY%%	$num_array 	"
}

mod_job < "$1" > "$3"
