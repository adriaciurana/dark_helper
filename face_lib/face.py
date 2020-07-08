class Face:
    def __init__(self, bb, character):
        self.bb = bb
        self.character = character

    def __expr__(self):
        return str(self)

    def __str__(self):
        return str({
            'name': self.character.name,
            'bb': self.bb,
        })

    @property
    def name(self):
        return self.character.name

    @property
    def thumbnail(self):
        return self.character.thumbnail

    @property
    def description(self):
        return self.character.description