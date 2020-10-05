from unittest import TestCase

from utils.component import BaseComponent


class TestBaseComponent(TestCase):
    def test_get_state_dict(self):
        bc = BaseComponent()

        sd = bc.state_dict()

        print(sd)
