import os
from typing import Iterable, Hashable


def get_file_paths(directory: str, name_filter: str) -> frozenset:
    """
    Get absolute file paths for files in a directory.
    A name filter is used to target the correct files (e.g. ".npy")
    """

    return frozenset(
        {
            os.path.join(directory, file_name)
            for file_name in os.listdir(directory)
            if name_filter in file_name
        }
    )


def get_file_paths_not_empty(file_paths: Hashable) -> frozenset:
    """Get only the non-empty files of a set of file paths"""

    return frozenset(filter(lambda x: os.path.getsize(x) > 0, file_paths))


def get_basedir(full_paths: Hashable) -> frozenset:
    """Get base directories from full paths"""

    return frozenset(map(os.path.basename, full_paths))


def take_common_set_elements(sets: Hashable) -> frozenset:
    """Intersect all sets"""

    sampled_set = next(iter(sets))
    assert type(sampled_set) != str
    return sampled_set.intersection(*sets)


def get_file_names_checked(directories: Hashable, name_filter: str) -> frozenset:
    """Use all checking pipeline to return file names to be loaded"""

    assert type(directories) != str

    file_names_to_compare = set()
    for directory in directories:
        # Load all file paths
        file_paths = get_file_paths(directory, name_filter)
        # Get only non-empty files
        checked_file_paths = get_file_paths_not_empty(file_paths)
        # Get only the base path to allow set comparison
        file_names = get_basedir(checked_file_paths)
        # Add to comparison set for intersection
        file_names_to_compare.add(frozenset(file_names))

    return frozenset(take_common_set_elements(file_names_to_compare))
