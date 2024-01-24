import json
import os
from typing import Tuple

from dotenv import load_dotenv
from telethon import functions
from telethon.errors import ChannelPrivateError, SessionPasswordNeededError
from telethon.sync import TelegramClient
from telethon.types import Channel, Chat, ChatFull, User

from analyzer import classify_text_by_theme, initialize_classification_model


def get_dotenv_data() -> Tuple[str, int, str]:
    load_dotenv()
    phone = os.getenv("PHONE") or ""
    api_id = int(os.getenv("API_ID") or 0)
    api_hash = os.getenv("API_HASH") or ""
    return phone, api_id, api_hash


def auth_client(client: TelegramClient):
    if not client.is_connected():
        client.connect()
    if not client.is_user_authorized():
        client.send_code_request(client.session.filename[:-8])
        code = input("code: ")
        try:
            client.sign_in(client._phone, code)
        except SessionPasswordNeededError:
            client.sign_in(password=input("password: "))
    return client


class Parser:
    def __init__(
        self,
        initial_entities: list,
        search_topic: str,
        search_depth: int,
        map_file: str,
    ):
        self.client = auth_client(TelegramClient(*get_dotenv_data()))
        self.entity_queue = initial_entities
        self.search_topic = search_topic
        self.search_depth = search_depth
        # self.chats_map = get_json_data(map_file)
        self.classifier_model = initialize_classification_model()

    def start(self):
        for entity in self.entity_queue:
            entity: User | Chat | Channel = self.client.get_entity(entity)
            print(entity.stringify())
            # Определяем тип сущности
            if type(entity) == Channel:
                full_channel: ChatFull = self.client(
                    functions.channels.GetFullChannelRequest(entity.username)
                )
                print(full_channel.stringify())
                total_texts = 0
                texts_topics = 0
                texts_not_topics = 0

                for message in self.client.iter_messages(entity):
                    total_texts += 1
                    print(message.text)
                    if message.raw_text:
                        result = classify_text_by_theme(
                            self.classifier_model,
                            message.raw_text,
                            self.search_topic,
                        )
                        print(result)
                        if result[0]:
                            texts_topics += 1
                        else:
                            texts_not_topics += 1
                        print(
                            total_texts,
                            texts_topics,
                            texts_not_topics,
                            texts_topics / (texts_not_topics + texts_topics),
                        )


if __name__ == "__main__":
    search_depth = 2
    initial_entities = [
        "https://t.me/usmfox_mining",
        "https://telegram.me/kreditniy_mining",
        "https://t.me/irk_miners",
        "https://telegram.me/mining_crypto_exchange",
        "https://t.me/china_mining_market",
        "https://t.me/Ric_mining",
        "https://t.me/nft_group2",
        "https://t.me/avitomining",
        "https://t.me/mining_applestore",
        "https://t.me/Maining_blockchain",
        "https://t.me/victorbavur",
        "https://t.me/miningmarket",
        "https://t.me/allminer_msk_chat",
        "https://t.me/MiningOnRussia",
        "https://t.me/nedomainer",
    ]
    search_topic = "майнинг"
    map_file = "map.json"
    parser = Parser(initial_entities, search_topic, search_depth, map_file)
    parser.start()
