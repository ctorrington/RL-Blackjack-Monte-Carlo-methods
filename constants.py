class Constants:

    class ACTIONS:
        HIT = "HIT"
        STICK = "STICK"

        @staticmethod
        def as_tuple():
            return Constants.ACTIONS.HIT, Constants.ACTIONS.STICK