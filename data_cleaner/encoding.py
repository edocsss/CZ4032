HEARD_OF_MAP = {
    'heard of': '1000000',
    'ever heard music by': '0100000',
    'ever heard of': '0010000',
    'heard of and listened to music ever': '0001000',
    'heard of and listened to music recently': '0000100',
    'listened to recently': '0000010',
    'never heard of': '0000001'
}

OWN_ARTIST_MAP = {
    'do not know': '10000',
    'own a little of their music': '01000',
    'own a lot of their music': '00100',
    'own all or most of their music': '00010',
    'own none of their music': '00001'
}

GENDER_MAP = {
    'male': '01',
    'female': '10'
}

WORKING_MAP = {
    'employed 30+ hours a week': '00000000000001',
    'employed 8-29 hours per week': '00000000000010',
    'employed part-time less than 8 hours per week': '00000000000100',
    'full-time housewife / househusband': '00000000001000',
    'full-time student': '00000000010000',
    'in unpaid employment (e.g. voluntary work)': '00000000100000',
    'other': '00000001000000',
    'part-time student': '00000010000000',
    'prefer not to state': '00000100000000',
    'retired from full-time employment (30+ hours per week)': '00001000000000',
    'retired from self-employment': '00010000000000',
    'self-employed': '00100000000000',
    'temporarily unemployed': '01000000000000',
    'not available': '10000000000000'
}

REGION_MAP = {
    'centre': '0000001',
    'midlands': '0000010',
    'north': '0000100',
    'north ireland': '0001000',
    'northern ireland': '0010000',
    'south': '0100000',
    'not available': '1000000'
}

MUSIC_MAP = {
    'i like music but it does not feature heavily in my life': '000001',
    'music has no particular interest for me': '000010',
    'music is important to me but not necessarily more important': '000100',
    'music is important to me but not necessarily more important than other hobbies or interests': '001000',
    'music is no longer as important as it used to be to me': '010000',
    'music means a lot to me and is a passion of mine': '100000'
}

HEARD_OF_MAP_REVERSED = {v: k for k, v in HEARD_OF_MAP.items()}
OWN_ARTIST_MAP_REVERSED = {v: k for k, v in OWN_ARTIST_MAP.items()}