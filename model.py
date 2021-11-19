# coding: utf-8
import pytz
from sqlalchemy import Column, Integer, String, TIMESTAMP, text
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime as dt

Base = declarative_base()
metadata = Base.metadata
tz = pytz.timezone('Asia/Taipei')


class Record(Base):
    __tablename__ = 'record'

    pk_id = Column(Integer, autoincrement=True, primary_key=True)
    user_id = Column(String(100, 'utf8mb4_unicode_ci'), nullable=False)
    session_id = Column(String(100, 'utf8mb4_unicode_ci'), nullable=False)
    req_sentence = Column(String(100, 'utf8mb4_unicode_ci'), nullable=False)
    resp_status = Column(TINYINT(1), nullable=False)
    resp_sentence = Column(String(500, 'utf8mb4_unicode_ci'))
    is_transfer = Column(TINYINT(1))
    trans_from = Column(String(100, 'utf8mb4_unicode_ci'))
    trans_to = Column(String(100, 'utf8mb4_unicode_ci'))
    amount = Column(String(100, 'utf8mb4_unicode_ci'))
    create_time = Column(TIMESTAMP, nullable=False, default=dt.now(tz), server_default=text("CURRENT_TIMESTAMP"))


class Account(Base):
    __tablename__ = 'account'

    pk_id = Column(Integer, autoincrement=True, primary_key=True)
    user_id = Column(String(100, 'utf8mb4_unicode_ci'), nullable=False)
    nickname = Column(String(100, 'utf8mb4_unicode_ci'), nullable=False)
    account_number = Column(String(25, 'utf8mb4_unicode_ci'), nullable=False)
    create_time = Column(TIMESTAMP, nullable=False, default=dt.now(tz), server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(TIMESTAMP, nullable=False, default=dt.now(tz))
