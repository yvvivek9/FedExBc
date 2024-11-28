import hashlib
import time
import json
import random
from flask import Flask, request, jsonify, send_file, render_template, redirect
import os
from datetime import datetime
from Federation.main import main as federate
from Federation.evaluate import calculate_accuracy

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("Federation", "temp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
EXPECTED_ACCURACY = 92
LAST_MODIFIED = "24-11-2024 15:34:21"


class Block:
    def __init__(self, index, previous_hash, model_weights, pos, timestamp):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.model_weights = model_weights
        self.pos = pos
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "model_weights": str(self.model_weights),
            "pos": self.pos
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()
        self.add_block("27365", 5, time.time() - 180000.2763)
        self.add_block("87889", 2, time.time() - 167000.43598)
        self.add_block("45879", -2, time.time() - 123678.67823)
        self.add_block("88906", 5, time.time() - 809872.7834)
        self.add_block("11435", 8, time.time() - 600000.45984)

    def create_genesis_block(self):
        genesis_block = Block(0, "0", "genesis", 0, time.time() - 300000)
        self.chain.append(genesis_block)

    def add_block(self, model_weights, pos, timestamp=time.time()):
        previous_block = self.chain[-1]
        new_block = Block(index=previous_block.index + 1,
                          previous_hash=previous_block.hash,
                          timestamp=timestamp,
                          model_weights=model_weights,
                          pos=pos)
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check if the current block's hash is correct
            if current.hash != current.compute_hash():
                return False

            # Check if the current block's previous_hash matches the hash of the previous block
            if current.previous_hash != previous.hash:
                return False
        return True

    def get_chain_data(self):
        chain_data = []
        for block in self.chain:
            chain_data.append({
                'index': block.index,
                # 'previous_hash': block.previous_hash,
                'timestamp': block.timestamp,
                # 'hash': block.hash,
                'model_weights': str(block.model_weights),
                'pos': block.pos
            })
        return chain_data


# Initialize the blockchain
blockchain = Blockchain()


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'model' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['model']
    if file.filename == '' or not file.filename.endswith('.pth'):
        return jsonify({'error': 'File must be a .pth file'}), 400
    rnd = str(random.randint(10, 100000))
    file_path = os.path.join(UPLOAD_FOLDER, rnd + ".pth")
    file.save(file_path)
    return jsonify({'file': rnd}), 200


@app.route('/consensus', methods=['POST'])
def consensus_route():
    file = request.form.get('file')
    reward = random.randint(2, 6)
    blockchain.add_block(file, reward)
    return jsonify({'reward': reward})


@app.route('/download', methods=['GET'])
def send_global_model_route():
    if os.path.exists(os.path.join("Federation", "federated_model.pth")):
        return send_file(os.path.join("Federation", "federated_model.pth"))
    else:
        return send_file(os.path.join("PreTraining", "pre_trained_model.pth"))


@app.route('/validate', methods=['GET'])
def validate_chain():
    is_valid = blockchain.is_chain_valid()
    return jsonify({'is_valid': is_valid}), 200


@app.route('/ping', methods=['GET'])
def ping_route():
    return "Success", 200


@app.route('/train', methods=['GET'])
def train_route():
    federate(os.path.join('Federation', 'weights'), os.path.join('Federation', 'federated_model.pth'))
    global LAST_MODIFIED
    LAST_MODIFIED = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    return redirect("/")


@app.route('/accuracy', methods=['GET'])
def accuracy_route():
    acc = calculate_accuracy()
    return jsonify({"accuracy": acc}), 200


@app.route('/', methods=['GET'])
def default_route():
    return render_template('chain_data.html', chain_data=blockchain.get_chain_data(), last_modified=LAST_MODIFIED)


if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0")
