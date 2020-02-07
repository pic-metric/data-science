#import psychopg2
from sqlalchemy import create_engine, select, update, insert
from sqlalchemy.ext.automap import automap_base
from dotenv import load_dotenv
import os
from base64 import b64decode, b64encode

load_dotenv()

engine = create_engine(os.getenv('DATABASE_URL'))

"""def query_one(query):
    curs = engine.connect()
    response = curs.execute(query).fetchone()
    curs.close()
    return response


def query_all(query):
    curs = engine.connect()
    response = curs.execute(query).fetchall()
    curs.close()
    return response"""


def get_image(image_id):
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    pics = Base.classes.pics
    query = select([pics.pic]).where(pics.id == image_id)
    curs = engine.connect()
    response = curs.execute(query).fetchone()[0]
    image = b64decode(response)
    curs.close()
    return response


def put_image(image_id, processed_image, atts):
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    pics = Base.classes.pics
    attributes = Base.classes.attributes
    curs = engine.connect()
    updt = update(pics).where(pics.id==image_id).values(processed_pic=processed_image)
    curs.execute(updt)
    for key, value in atts.items():
        ins = insert(attributes).values(
            pic_id=image_id,
            attribute=key,
            count=value)
        curs.execute(ins)
    curs.close()


"""def produce_test_image(image_path):
    with open(image_path, 'rb') as file:
        text = file.read()
        return text"""


"""def query(query):
    conn = engine.connect()
    conn.execute(query)
    conn.close()"""