import sqlite3
import os
import enum

"""Data is structured in two tables:
    datas: 
        id: integer primary key,
        ot: integer,
        alias: integer,
        setcode: integer,
        type: integer,
        atk: integer,
        def: integer,
        level: integer,
        race: integer,
        attribute: integer,
        category: integer.
        
    texts:
        id: integer primary key,
        name: text,
        desc: text,
        str1: text,
        str2: text,
        ......
        str16: text.
"""


class CardTypes(enum.Enum):
    MONSTER = 1
    SPELL = 2
    TRAP = 4


class CardRaces(enum.Enum):
    WARRIOR = 1
    SPELLCASTER = 2
    FAIRY = 4
    FIEND = 8
    ZOMBIE = 16
    MACHINE = 32
    AQUA = 64
    PYRO = 128
    ROCK = 256
    WINGEDBEAST = 512
    PLANT = 1024
    INSECT = 2048
    THUNDER = 4096
    DRAGON = 8192
    BEAST = 16384
    BEASTWARRIOR = 32768
    DINOSAUR = 65536
    FISH = 131072
    SEASERPENT = 262144
    REPTILE = 524288
    PSYCHIC = 1048576
    DIVINE = 2097152
    CREATORGOD = 4196304
    WYRM = 8388608
    CYBERSE = 16777216
    YOKAI = 2147483648
    CHARISMA = 4294967296


class CardInfo:
    def __init__(self, data):
        self.data = data

    def getID(self):
        return self.data[0]

    def getType(self):
        return self.data[4]

    def getATK(self):
        return self.data[5]

    def getDEF(self):
        return self.data[6]

    def getLevel(self):
        return self.data[7]

    def getRace(self):
        return self.data[8]

    def getAttribute(self):
        return self.data[9]

    def getCategory(self):
        return self.data[10]

    def getName(self):
        return self.data[11]

    # def getDescription(self):
    #     return self.fromTexts(2)


class CardReader(object):
    def __init__(self):
        self.id = None
        con = sqlite3.connect(os.path.join('resources', 'card_utils', 'cards.cdb'))
        self.cur = con.cursor()

    def setID(self, ID):
        self.id = ID

    def fromData(self, n):
        self.cur.execute("SELECT * FROM datas WHERE id = {}".format(self.id))
        return self.cur.fetchall()[0][n]

    def fromTexts(self, n):
        self.cur.execute("SELECT * FROM texts WHERE id = {}".format(self.id))
        return self.cur.fetchall()[0][n]

    def get_all(self, key, value):
        self.cur.execute("SELECT * FROM datas WHERE {} = {}".format(key, value))
        data_list = self.cur.fetchall()
        return [CardInfo(data) for data in data_list]

    def get_all_flag(self, key, flag):
        if isinstance(flag, enum.Enum):
            flag = flag.value
        self.cur.execute("SELECT * FROM datas WHERE {} & {} != 0".format(key, flag))
        data_list = self.cur.fetchall()
        return [CardInfo(data) for data in data_list]

    def getType(self):
        return self.fromData(4)
    
    def getATK(self):
        return self.fromData(5)
    
    def getDEF(self):
        return self.fromData(6)
    
    def getLevel(self):
        return self.fromData(7)
    
    def getRace(self):
        return self.fromData(8)
    
    def getAttribute(self):
        return self.fromData(9)
    
    def getCategory(self):
        return self.fromData(10)
    
    def getName(self):
        return self.fromTexts(1)

    def getDescription(self):
        return self.fromTexts(2)


if __name__ == '__main__':

    from utils.dir import Dir
    Dir.set_project_dir()


    info = CardReader()
    cards = info.get_all_flag('type', CardTypes.MONSTER)
    print(cards)

