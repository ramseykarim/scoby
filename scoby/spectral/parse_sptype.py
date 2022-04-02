"""
also will contain functions for parsing the catalog's spectral types
file used to be called readstartypes.py
Created: November 11, 2019

I am returning to this code to treat uncertainties and binaries properly
Wow there's a lot of regex in here, good job past Ramsey
Revisited: April 29-30, 2020

Revisited: June 1, 2020
June 2, 2020: Split apart in to several files
June 11, 2020: Updated subtype slash behavior

Revisited: August 4, 2020
I edited st_parse_slashdash to completely ignore colons ":", which
are used to denote uncertainty. I could change this in the future, but the
intention of the author is not always clear.
"""
__author__ = "Ramsey Karim"

import re

luminosity_classes = ('V', 'III', 'I')
# search for: standard types, WR, Herbig Ae/Be (type of PMS), and pre main sequence
nonstandard_types_re = '(W(N|C))|(HAeBe)|(PMS)|(C)'
standard_types = "OBAFGKM"
standard_types_re = f'[{standard_types}]'
letter_re = f'({standard_types_re}|{nonstandard_types_re})'
roman_num_re = '(I+|I?V)'
peculiar_re = '((\\+|(ha)|n|(\\(*(f|e)(\\+|\\*)?\\)*))+\\*?)'
number_re = '(\\d{1}(\\.\\d)?)'
slashdash_re = '(/|-)'

INVALID_STAR_FLAG = float('NaN')


"""
===============================================================================
================== Spectral Type parsing via regex ============================
===============================================================================
"""


def re_parse_helper(pattern, string):
    """
    Return match string if it exists, else empty string.
    Not always advantageous over just doing re.search, but shorter in some
    cases.
    :param pattern: regex-ready string pattern
    :param string: the string to search
    :returns: str; match if exists, otherwise ''
    """
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return ''


def st_parse_type(spectral_type_string):
    """
    Parse a single full class string (no slashes or dashes)
    i.e. O5.5III((f))* => 'O', '5.5', 'III', '((f))*'
    :param spectral_type_string: string descrbing spectral type
        No uncertainty ('/' or '-') and no binaries ('+')
        Peculiarities are ok
    :returns: tuple(string), set up like:
        (letter, number, luminosity_class, peculiarity)
    """
    pec = re_parse_helper(peculiar_re, spectral_type_string)
    lumclass = re_parse_helper(roman_num_re, spectral_type_string)
    subtype = re_parse_helper(number_re, spectral_type_string)
    lettertype = re_parse_helper(letter_re, spectral_type_string)
    return (lettertype, subtype, lumclass, pec)


def st_parse_slashdash(spectral_type_string, intercept_WR=True):
    """
    Take a spectral type string and return a list of all uncertain
        possibilities based on that string's use of dashes and slashes
    i.e. O3-4V => ['O3V', 'O3.5V', 'O4V'], or O4-5.5III/If => 8 options
    Peculiarities are ok. Just no "+".
    :param spectral_type_string: string descrbing spectral type
        No '+' signs, cannot be binary
        Peculiarities are ok
    :param intercept_WR: this is an (admittedly hardcodey) switch to
        intercept "slash" types (like O2If/WN5) and assign the WR type (WN5 in
        that example). The purpose of this is to be more honest about the type
        of physical star these actually are. This function interprets slashes
        as uncertainties, but in these "slash star" types, it's part of the
        type designation.
    :returns: list(string) where strings are possible spectral types

    June 11, 2020:
    Updated "dash" subclass behavior to be expressed as a range of types instead
        of behaving as an alias for "slash". This will be better for MC sampling
        spectral type error.
        Note that dashed luminosity class behave like slashes because we don't
        have totally smooth model coverage of luminosity classes.
    June 17, 2020:
    Updated to allow dashes between letter classes, e.g. O9-B3, with behavior
        as described above.
    August 4, 2020:
    Updated to completely ignore ":", which denotes uncertainty.
    """
    # First, replace ":" with an empty string; completely ignore ":"
    spectral_type_string = spectral_type_string.replace(':', '')
    if not re.search(slashdash_re, spectral_type_string):
        # Check for / and return list(string) if not
        return [spectral_type_string]
    # Parse the slash notation from SINGLE star (not binary)
    # The O/WN stars are a confusing category: see Wikipedia Wolf-Rayet stars: Slash stars
    elif '/' in spectral_type_string and spectral_type_string[spectral_type_string.index('/') + 1] == 'W':
        # This is a "slash star"
        if intercept_WR:
            # Return the WR type; assume no other slashes
            return [spectral_type_string.split('/')[1]]
        else:
            # Return both complete spectral types
            return spectral_type_string.split('/')
    # Add possible luminosity classes to this list
    possible_lumclasses = []
    pattern_lumclass = f'{roman_num_re}{peculiar_re}?({slashdash_re}{roman_num_re}{peculiar_re}?)+'
    match_lumclass = re.search(pattern_lumclass, spectral_type_string)
    if match_lumclass:
        # print("-----", spectral_type_string, '->', match_lumclass.group())
        possible_lumclasses.extend(match_lumclass.group().replace('-', '/').split('/'))
        for i in range(len(possible_lumclasses)):
            possible_lumclasses[i] = spectral_type_string.replace(match_lumclass.group(), possible_lumclasses[i])
    else:
        possible_lumclasses.append(spectral_type_string)
    possible_classes = []
    # Break down class/subclass cases now that each only has one luminosity class
    for possible_lumclass in possible_lumclasses:
        # Slash or dash in subclass number
        pattern_subclass = f'{number_re}({slashdash_re}{number_re})+'
        match_subclass = re.search(pattern_subclass, possible_lumclass)
        # Slash or dash across standard letter types
        pattern_crossclass = f'{standard_types_re}{number_re}({slashdash_re}{standard_types_re}{number_re})+'
        match_crossclass = re.search(pattern_crossclass, possible_lumclass)
        # Run through the cases
        if match_subclass:
            # Case that only the subclass (number) has a slash/dash
            # Separate dash- from slash/ behavior
            if '-' in match_subclass.group():
                # Dash
                # Numbers should all be multiples of 0.5
                subclass_bounds = [int(float(x)*2) for x in match_subclass.group().split('-')]
                # There should be at most one dash
                assert len(subclass_bounds) == 2
                subclass_options = [x/2. for x in range(subclass_bounds[0], subclass_bounds[1]+1)]
                for sc_option in subclass_options:
                    sc_option = str(sc_option).replace('.0', '')
                    possible_classes.append(possible_lumclass.replace(match_subclass.group(), sc_option))
            else:
                # Slash (simpler case)
                possible_subclasses = match_subclass.group().split('/')
                for i in range(len(possible_subclasses)):
                    possible_classes.append(possible_lumclass.replace(match_subclass.group(), possible_subclasses[i]))
        elif match_crossclass:
            # Case that there is a slash or dash across letter types
            # Separate dash and slash behavior
            if '-' in match_crossclass.group():
                # Dash; convert everything to numbers. Should be multiples of 0.5
                class_bounds = [int(st_to_number(x)*2) for x in match_crossclass.group().split('-')]
                # There should be at most one dash
                assert len(class_bounds) == 2
                class_options = [x/2. for x in range(class_bounds[0], class_bounds[1]+1)]
                for c_option in class_options:
                    c_option = ''.join(number_to_st(c_option))
                    possible_classes.append(possible_lumclass.replace(match_crossclass.group(), c_option))
            else:
                # Slash; simpler case
                # I originally wrote this on April 30, 2020; cleaned up on June 17, 2020
                possible_classes.extend(possible_lumclass.split('/'))
        else:
            # No match, just assume only one possibility
            possible_classes.append(possible_lumclass)
    return possible_classes


def st_parse_binary(spectral_type_string):
    """
    Identify binaries (+)
    :param spectral_type_string: string describing spectral type
    :returns: list(string) where strings are binary components
    """
    if spectral_type_string[-1] == '+':
        return [spectral_type_string]
    else:
        return spectral_type_string.split('+')



def st_tuple_to_string(t):
    """
    Join up to 3 strings in a (spectral type) tuple
    :param t: the tuple of strings
    :returns: str; joined from tuple
    """
    return "".join(t[:3])


def st_to_number(spectral_type):
    """
    Returns a float number based on the spectral type, increasing with later type
    Wow ok this exact system was used in Vacca et al 1996... nice one dude
    :param spectral_type: Just simple letter+number spectral_type,
        decminal optional.
        Like "O3" or "B1.5"
        OR tuple ('letter', 'number', 'lumclass', 'peculiarity')
    :returns: float number, or NaN if cannot make conversion
    """
    if isinstance(spectral_type, str):
        if re.search(nonstandard_types_re, spectral_type):
            return INVALID_STAR_FLAG
        t = re.search(standard_types_re, spectral_type).group()
        subt = re.search(number_re, spectral_type).group()
    else:
        t, subt = spectral_type[0], spectral_type[1]
    if (not t) or (not subt):
        return INVALID_STAR_FLAG
    elif t not in standard_types:
        return INVALID_STAR_FLAG
    else:
        return standard_types.index(t)*10. + float(subt)


def number_to_st(spectral_type_number):
    """
    Written: April 29, 2020
    Reverse of st_to_number
    Returns the tuple expression of spectral type; type and subtype only
        e.g. 11.5 -> ('B', '1.5')
    """
    t = int(spectral_type_number//10)
    subt = spectral_type_number % 10
    return (standard_types[t], f"{subt:.1f}".replace(".0", ""))


def lc_to_number(lumclass):
    if isinstance(lumclass, tuple):
        lumclass = lumclass[2]
    if not lumclass:
        # Edited April 29, 2020: I used to return INVALID_STAR_FLAG here,
        #   but now I'm going to assign 'V' because that makes more sense
        lumclass = 'V'
    return ['I', 'II', 'III', 'IV', 'V'].index(lumclass) + 1


def st_adjacent(spectral_type_tuple):
    """
    For a given standard spectral type tuple, returns 3 similar tuples:
    A half type earlier, the original argument type, and a half type later.
    This should be representative of reasonable half-type uncertainty, which
        should apply to both single spectral types as well as statedly uncertain
        types.
    If the input tuple is more than 2 elements, the third and further elements
        are simply duplicated into the adjacent type tuples.
    :param spectral_type_tuple: standard tuple format
    :returns: list of spectral type tuples, increasing (later) type
    """
    spectral_type_number = st_to_number(spectral_type_tuple)
    earlier, later = spectral_type_number - 0.5, spectral_type_number + 0.5
    earlier, later = number_to_st(earlier), number_to_st(later)
    if len(spectral_type_tuple) > 2:
        earlier = earlier + spectral_type_tuple[2:]
        later = later + spectral_type_tuple[2:]
    return [earlier, spectral_type_tuple, later]


def st_reduce_to_brightest_star(st):
    """
    Reduce a full spectral type to the "brightest" possibility + binary
        component.
    NOTE: this function should be deleted; it is not useful anymore and
        it reduces too much so the results are unphysical and very approximate.
    TODO: delet this
    """
    # st = string
    st = st_parse_binary(st)
    # st = list(string)
    st = [st_parse_slashdash(x) for x in st]
    # st = list(list(string))
    st = [[st_parse_type(y) for y in x] for x in st]
    # st = list(list(tuple(string)))
    st = [min(x, key=st_to_number) for x in st]
    # st = list(tuple(string))
    st = min(st, key=st_to_number)
    # st = tuple(string)
    return st


def sanitize_tuple(spectral_type_tuple):
    """
    Input sanitization, used in STTables
    Returns False if the spectral type isn't "standard" OBAFGKM
    Assigns luminosity class V if not specified
    Returns spectral type tuple that is sanitized
    """
    if re.search(nonstandard_types_re, spectral_type_tuple[0]):
        return False
    if len(spectral_type_tuple) == 2:
        return spectral_type_tuple + ('V',)
    elif not re.search(roman_num_re, spectral_type_tuple[2]):
        spectral_type_tuple = list(spectral_type_tuple)
        spectral_type_tuple[2] = 'V'
        return tuple(spectral_type_tuple)
    return spectral_type_tuple


def reduce_catalog_spectral_types(cat):
    """
    Created: Unsure, probably November 2019
    Reviewed April 29, 2020
    This seems to be where I make a lot of my assumptions and boil down the
        spectral types, getting rid of binaries and glossing over uncertainties
        (ranges of possible types)
    June 2, 2020: I think we can get rid of this function.
    TODO: delet this
    """
    cat['SpectralType_Adopted'] = cat.SpectralType.where(cat.SpectralType != 'ET', other='O9V', inplace=False)
    cat['SpectralType_ReducedTuple'] = cat.SpectralType_Adopted.apply(st_reduce_to_brightest_star)
    cat['SpectralType_Reduced'] = cat.SpectralType_ReducedTuple.apply(st_tuple_to_string)
    cat['SpectralType_Number'] = cat.SpectralType_Reduced.apply(st_to_number)
