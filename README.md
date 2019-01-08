#  说明：

5个class，1把锁chain_lock，然后定义了一组函数。

这一组函数粗略地划分了若干个模块，# chain，# Chain Persistance，# UTXO set， # Proof of work， # Validation，# mempool， # Merkle trees， # Peer-to-peer，# Wallet， # Misc. utilities。

### 目的： 
划分几个模块，使得功能单一化、松耦合化，每个模块写在一个单独的Python文件里。以后对某个功能进行开发扩展时，只要考虑对应的模块，而不用太过于考虑其它模块的代码。

### 另外：
  +  该Python代码的中文注释： https://github.com/EdgeIntelligenceChain/EdgenceChain/blob/comments_by_jinliang/edgencechain.py
  +  区块链和比特币的基础知识： https://github.com/EdgeIntelligenceChain/Materials2Study


# Edgencechain



Edgencechain is a pocket-sized implementation of Bitcoin. Its goal is to
be a compact, understandable, working incarnation of 
[the Nakamoto consensus algorithm](https://github.com/EdgeIntelligenceChain/Materials2Study/blob/master/%E6%AF%94%E7%89%B9%E5%B8%81%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) at the
expense of advanced functionality, speed, and any real usefulness.

## 代码文件

- 区块链核心代码

  | 文件名 | 内容 | 
  | :------| :------ |
  | edgencechain | 区块链核心 | 
  | ├ chain.py | 区块链核心 |
  | ├ definitions.py | 数据结构定义 |
  | ├ helpers.py | 工具函数 |
  | ├ params.py | 参数 |
  | ├ wallet.py | 钱包相关 |
  | ├ mempool.py | |
  | client.py | 客户端 |
 
- Docker配置文件

  | 文件名 | 内容 | 
  | :------| :------ |
  | Dockerfile | 创建容器的镜像 | 
  | docker-compose.yaml | 生成节点及其连接的设置 |
  | requirements.txt | 为Dockerfile所用 |
  | bin/sync_wallets|将容器中的wallet数据转出到本机|
  | start-docker.sh|Linux脚本，启动Docker|


- 测试代码

  | 文件名 | 内容 | 
  | :------| :------ |
  | test_edgencechain.py | 测试代码 | 
  | test.sh | Linux脚本，启动测试代码 |
  | requirements.test.txt|测试脚本的输入|


## 快速上手

- Ubuntu 16.04及以上版本，Python3.6.2及以上版本
- Install Docker & docker-compose
  + [Docker CE安装](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/)
  + [Docker Compose安装](https://docs.docker.com/compose/install/#install-compose)


- Clone this repo: `git clone git@github.com:EdgeIntelligenceChain/edgencechain.git`
- Make sure you're in a Python3.6 environment: `sudo virtualenv --python=python3.6 venv && . venv/bin/activate`
- Grab Python dependencies locally: `sudo pip install -r requirements.txt`
- Run `sudo docker-compose up`. This will spawn two edgencechain nodes.
- In another window, run `sudo ./bin/sync_wallets`. This brings the wallet data
  from the Docker containers onto your host.
    ```
    $ sudo ./bin/sync_wallets

    Synced node1's wallet:
    [2017-08-05 12:59:34,423][edgencechain:1075] INFO your address is 1898KEjkziq9uRCzaVUUoBwzhURt4nrbP8
     0.0 ⛼

    Synced node2's wallet:
    [2017-08-05 12:59:35,876][edgencechain:1075] INFO your address is 15YxFVo4EuqvDJH8ey2bY352MVRVpH1yFD
    0.0 ⛼
    ```
- Try running `sudo ./client.py balance -w wallet1.dat`; try it with the other
  wallet file.
    ```
    $ ./client.py balance -w wallet2.dat

    [2017-08-05 13:00:37,317][edgencechain:1075] INFO your address is 15YxFVo4EuqvDJH8ey2bY352MVRVpH1yFD
    0.0 ⛼
    ```
- Once you see a few blocks go by, try sending some money between the wallets
    ```
    $ ./client.py send -w wallet2.dat 1898KEjkziq9uRCzaVUUoBwzhURt4nrbP8 1337
    
    [2017-08-05 13:08:08,251][edgencechain:1077] INFO your address is 1Q2fBbg8XnnPiv1UHe44f2x9vf54YKXh7C
    [2017-08-05 13:08:08,361][client:105] INFO built txn Transaction(...)
    [2017-08-05 13:08:08,362][client:106] INFO broadcasting txn 2aa89204456207384851a4bbf8bde155eca7fcf30b833495d5b0541f84931919
    ```
- Check on the status of the transaction
    ```
     $ ./client.py status e8f63eeeca32f9df28a3a62a366f63e8595cf70efb94710d43626ff4c0918a8a

     [2017-08-05 13:09:21,489][edgencechain:1077] INFO your address is 1898KEjkziq9uRCzaVUUoBwzhURt4nrbP8
     Mined in 0000000726752f82af3d0f271fd61337035256051a9a1e5881e82d93d8e42d66 at height 5
    ```


## 比特币简介

In brief terms that map to this code...

Bitcoin is a way of generating pseudo-anonymous, decentralized trust at the cost
of electricity. The most commonly known (but not sole) application of this is as
a currency or store of value. If that sounds abstruse, general, and mindblowing,
that's because it is.

In Bitcoin, value is recorded using a `Transaction`, which assigns some
number of coins to an identity (via `TxOut`s) given some cryptographically
unlocked `TxIn`s.  TxIns must always refer to previously created but unspent
TxOuts.

A Transaction is written into history by being included in a `Block`. Each Block
contains a data structure called a [Merkle
Tree](https://en.wikipedia.org/wiki/Merkle_tree) which generates a fingerprint
unique to the set of Transactions being included. The root of that Merkle tree
is included in the block "header" and hashed (`Block.id`) to permanently seal
the existence and inclusion of each Transaction in the block.

Blocks are linked together in a chain (`active_chain`) by referring to the
previous Block header hash. In order to add a Block to the chain, the contents
of its header must hash to a number under some difficulty target, which is set
based upon how quickly recent blocks have been discovered
(`get_next_work_required()`). This attempts to
normalize the time between block discovery.

When a block is discovered, it creates a subsidy for the discoverer in the form
of newly minted coins. The discoverer also collects fees from transactions
included in the block, which are the value of inputs minus outputs. The block
reward subsidy decreases logarithmically over time. Eventually the subsidy 
goes to zero and miners are incentivized to continue mining purely by a fee
market.

Nodes in the network are in a never-ending competition to mine and propagate the
next block, and in doing so facilitate the recording of transaction history.
Transactions are submitted to nodes and broadcast across the network, stored
temporarily in `mempool` where they are queued for block inclusion.



  
## 不同于比特币之处
 
- Byte-level representation and endianness are very important when serializing a
  data structure to be hashed in Bitcoin and are not reproduced
  faithfully here. In fact, serialization of any kind here is very dumbed down
  and based entirely on raw strings or JSON.

- Transaction types are limited to pay-to-public-key-hash (P2PKH), which
  facilitate the bare minimum of "sending money." More exotic
  [transaction
  types](https://bitcoin.org/en/developer-guide#standard-transactions) which 
  allow m-of-n key signatures and
  [Script](https://en.bitcoin.it/wiki/Script)-based unlocking are not
  implemented.

- [Initial Block Download](https://bitcoin.org/en/developer-guide#initial-block-download) 
  is at best a simplified version of the old "blocks-first" scheme. It eschews 
  `getdata` and instead returns block payloads directly in `inv`.

- The longest, valid chain is determined simply by chain length (number of
  blocks) vs. [chainwork](https://bitcoin.stackexchange.com/questions/26869/what-is-chainwork).

- Peer "discovery" is done through environment variable hardcoding. In
  bitcoin core, this is done [with DNS seeds](https://en.bitcoin.it/wiki/Transaction_replacement).

- [Replace by fee](https://en.bitcoin.it/wiki/Transaction_replacement) is absent.

- Memory usage is egregious. Networking is a hack.


 

## Q&A

### How does RPC work?

We use JSON for over-the-wire serialization. It's slow and unrealistic but
human-readable and easy. We deserialize right into the `.*Msg` classes, 
each of which dictates how a particular RPC message is handled via 
`.handle()`.

### Why doesn't the client track coins we've spent but haven't confirmed yet?

Yeah I know, the client sucks. I'll take a PR.

### How can I add another RPC command to reveal more data from a node?

Just add a `NamedTuple` subclass with a `handle()` method defined; it registers
automatically. Mimic any existing `*Msg` class.

 
### Why aren't my changes changing anything?

Remember to rebuild the Docker container image when you make changes
```
docker-compose build && docker-compose up
```

### How do I run automated tests?

```
pip install -r requirements.test.txt
py.test --cov test_edgencechain.py
```






