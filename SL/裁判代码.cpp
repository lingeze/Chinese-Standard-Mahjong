#include<iostream>
#include<sstream>
#include<string>
#include<cstdlib>
#include<unordered_map>
#include<algorithm>
#include"jsoncpp/json.h"
#include"MahjongGB/fan_calculator.cpp"
#include"MahjongGB/shanten.cpp"

using namespace std;

unordered_map<string, mahjong::tile_t> str2tile;

Json::Value inputObj, outputObj;
int seed, quan;
string walltiles;
vector<string> tilewall;

unordered_map<string, int> showntiles;
int state, curplayer;
string curtile;
bool isaboutkong, drawaboutkong, walllast;
int canHu[4];

enum MeldType {
	CHOW, PUNG, KONG
};

struct Meld {
	MeldType type;
	uint8_t offer;
	string tile;
};

struct {
	vector<Meld> melds;
	vector<string> hand;
	vector<string> tilewall;
} players[4];

void split(vector<string> &v, const string &s) {
	istringstream iss(s);
	string t;
	v.clear();
	while(iss >> t) v.push_back(t);
}

void init() {
	for(int i = 1; i <= 9; ++i) {
		str2tile["W" + to_string(i)] = mahjong::make_tile(TILE_SUIT_CHARACTERS, i);
		str2tile["B" + to_string(i)] = mahjong::make_tile(TILE_SUIT_DOTS, i);
		str2tile["T" + to_string(i)] = mahjong::make_tile(TILE_SUIT_BAMBOO, i);
	}
	for(int i = 1; i <= 4; ++i) {
		str2tile["F" + to_string(i)] = mahjong::make_tile(TILE_SUIT_HONORS, i);
	}
	for(int i = 1; i <= 3; ++i) {
		str2tile["J" + to_string(i)] = mahjong::make_tile(TILE_SUIT_HONORS, i + 4);
	}
}

void initdata() {
	Json::Value data = inputObj["initdata"];
	seed = time(nullptr);
	if(data.isObject() && data["srand"].isInt())
		seed = data["srand"].asInt();
	srand(seed);
	quan = -1;
	if(data.isObject() && data["quan"].isInt())
		quan = data["quan"].asInt();
	if(quan < 0 || quan > 3) quan = rand() % 4;
	if(data.isObject() && data["walltiles"].isString()) {
		walltiles = data["walltiles"].asString();
		split(tilewall, walltiles);
		if(tilewall.size() != 136) walltiles.clear();
		else {
			unordered_map<string, int> count;
			for(auto &s : tilewall) ++count[s];
			for(auto &p : str2tile)
				if(count[p.first] != 4) {
					walltiles.clear();
					break;
				}
		}
	}
	if(walltiles.empty()) {
		for(int i = 1; i <= 9; i++) {
			for(int j = 0; j < 4; j++) {
				tilewall.push_back("W" + to_string(i));
				tilewall.push_back("B" + to_string(i));
				tilewall.push_back("T" + to_string(i));
			}
		}
		for(int i = 1; i <= 4; i++) {
			for(int j = 0; j < 4; j++) {
				tilewall.push_back("F" + to_string(i));
			}
		}
		for(int i = 1; i <= 3; i++) {
			for(int j = 0; j < 4; j++) {
				tilewall.push_back("J" + to_string(i));
			}
		}
		random_shuffle(tilewall.begin(), tilewall.end());
		for(auto &s : tilewall) {
			walltiles += s;
			walltiles += ' ';
		}
		walltiles.pop_back();
	}
	int nw = tilewall.size() / 4;
	for(int i = 0; i < 4; ++i)
		for(int j = 0; j < nw; ++j)
			players[i].tilewall.push_back(tilewall[i * nw + j]);
	for(int i = 0; i < 4; ++i)
		for(int j = 0; j < 13; ++j) {
			string tile = *players[i].tilewall.rbegin();
			players[i].hand.push_back(tile);
			players[i].tilewall.pop_back();
		}
}

void initoutput() {
	outputObj["initdata"]["srand"] = seed;
	outputObj["initdata"]["quan"] = quan;
	outputObj["initdata"]["walltiles"] = walltiles;
	for(int i = 0; i < 4; ++i)
		outputObj["content"][to_string(i)] = "0 " + to_string(i) + " " + to_string(quan);
	outputObj["display"]["action"] = "INIT";
	outputObj["display"]["srand"] = seed;
	outputObj["display"]["quan"] = quan;
}

void erroroutput(int p, string code) {
	outputObj["command"] = "finish";
	outputObj["display"]["action"] = code;
	outputObj["display"]["player"] = p;
	for(int i = 0; i < 4; ++i) {
		outputObj["display"]["score"][i] = i == p ? -30 : 10;
		outputObj["content"][to_string(i)] = i == p ? -30 : 10;
	}
}

void tieoutput() {
	outputObj["command"] = "finish";
	outputObj["display"]["action"] = "HUANG";
	for(int i = 0; i < 4; ++i)
		outputObj["content"][to_string(i)] = 0;
}

int checkmahjong(int p, bool isselfdrawn, bool isaboutkong, bool finish = true) {
	mahjong::calculate_param_t calculate_param{};
	mahjong::fan_table_t fan_table{};
	int n = players[p].hand.size();
	calculate_param.hand_tiles.tile_count = n;
	for(int i = 0; i < n; ++i)
		calculate_param.hand_tiles.standing_tiles[i] = str2tile[players[p].hand[i]];
	n = players[p].melds.size();
	calculate_param.hand_tiles.pack_count = n;
	for(int i = 0; i < n; ++i) {
		switch(players[p].melds[i].type) {
			case CHOW:
				calculate_param.hand_tiles.fixed_packs[i] = mahjong::make_pack(players[p].melds[i].offer + 2, PACK_TYPE_CHOW, str2tile[players[p].melds[i].tile]);
				break;
			case PUNG:
				calculate_param.hand_tiles.fixed_packs[i] = mahjong::make_pack((players[p].melds[i].offer - p + 4) % 4, PACK_TYPE_PUNG, str2tile[players[p].melds[i].tile]);
				break;
			case KONG:
				calculate_param.hand_tiles.fixed_packs[i] = mahjong::make_pack((players[p].melds[i].offer - p + 4) % 4, PACK_TYPE_KONG, str2tile[players[p].melds[i].tile]);
				break;
		}
	}
	calculate_param.win_tile = str2tile[curtile];
	calculate_param.flower_count = 0;
	if(isselfdrawn) calculate_param.win_flag |= WIN_FLAG_SELF_DRAWN;
	if(walllast) calculate_param.win_flag |= WIN_FLAG_WALL_LAST;
	if(showntiles[curtile] + isselfdrawn == 4) calculate_param.win_flag |= WIN_FLAG_4TH_TILE;
	if(isaboutkong) calculate_param.win_flag |= WIN_FLAG_ABOUT_KONG;
	calculate_param.prevalent_wind = (mahjong::wind_t) quan;
	calculate_param.seat_wind = (mahjong::wind_t) p;
	int re = mahjong::calculate_fan(&calculate_param, &fan_table);
	if(finish) {
		outputObj["command"] = "finish";
		outputObj["display"]["action"] = "HU";
		outputObj["display"]["player"] = p;
		outputObj["display"]["fanCnt"] = re;
		for(int i = 0; i < mahjong::FAN_TABLE_SIZE; ++i)
			if(fan_table[i]) {
				Json::Value f;
				f["name"] = mahjong::fan_name[i];
				f["cnt"] = fan_table[i];
				f["value"] = mahjong::fan_value_table[i];
				outputObj["display"]["fan"].append(f);
			}
		if(re < 8) erroroutput(p, "WH");
		else for(int i = 0; i < 4; ++i)
			if(isselfdrawn) outputObj["content"][to_string(i)] = outputObj["display"]["score"][i] = i == p ? (re + 8) * 3 : -8 - re;
			else outputObj["content"][to_string(i)] = outputObj["display"]["score"][i] = i == p ? re + 24 : i == curplayer ? -8 - re : -8;
	}
	return re;
}

void draw(int p, bool check) {
	curtile = *players[p].tilewall.rbegin();
	players[p].tilewall.pop_back();
	walllast = players[(p + 1) % 4].tilewall.empty();
	isaboutkong = drawaboutkong;
	drawaboutkong = false;
	state = 1;
	if(check) {
		canHu[p] = checkmahjong(p, true, isaboutkong, false);
		for(int i = 0; i < 4; ++i)
			outputObj["content"][to_string(i)] = i == p ? "2 " + curtile : "3 " + to_string(p) + " DRAW";
		outputObj["display"]["action"] = "DRAW";
		outputObj["display"]["player"] = p;
		outputObj["display"]["tile"] = curtile;
	}
}

void discard(int p, string tile, bool check) {
	auto it = find(players[p].hand.begin(), players[p].hand.end(), tile);
	if(check)
		if(it == players[p].hand.end()) {
			erroroutput(p, "WA");
			return;
		}
	players[p].hand.erase(it);
	++showntiles[tile];
	walllast = players[(p + 1) % 4].tilewall.empty();
	curtile = tile;
	state = 2;
	if(check) {
		for(int i = 0; i < 4; ++i)
			if(i != p)
				canHu[i] = checkmahjong(i, false, false, false);
		for(int i = 0; i < 4; ++i)
			outputObj["content"][to_string(i)] = "3 " + to_string(p) + " PLAY " + curtile;
		outputObj["display"]["action"] = "PLAY";
		outputObj["display"]["player"] = p;
		outputObj["display"]["tile"] = curtile;
	}
}

void chow(int p, string midtile, string discard, bool check) {
	players[p].hand.push_back(curtile);
	--showntiles[curtile];
	if(check)
		if(midtile[0] != 'W' && midtile[0] != 'T' && midtile[0] != 'B') {
			erroroutput(p, "WA");
			return;
		}
	for(int i = -1; i < 2; ++i) {
		midtile[1] += i;
		++showntiles[midtile];
		auto it = find(players[p].hand.begin(), players[p].hand.end(), midtile);
		if(it == players[p].hand.end()) {
			erroroutput(p, "WA");
			return;
		}
		players[p].hand.erase(it);
		midtile[1] -= i;
	}
	players[p].melds.push_back({CHOW, curtile[1] - midtile[1], midtile});
	curplayer = p;
	auto it = find(players[p].hand.begin(), players[p].hand.end(), discard);
	if(check)
		if(it == players[p].hand.end()) {
			erroroutput(p, "WA");
			return;
		}
	players[p].hand.erase(it);
	++showntiles[discard];
	walllast = players[(p + 1) % 4].tilewall.empty();
	curtile = discard;
	if(check) {
		for(int i = 0; i < 4; ++i)
			if(i != p)
				canHu[i] = checkmahjong(i, false, false, false);
		for(int i = 0; i < 4; ++i)
			outputObj["content"][to_string(i)] = "3 " + to_string(p) + " CHI " + midtile + " " + discard;
		outputObj["display"]["action"] = "CHI";
		outputObj["display"]["player"] = p;
		outputObj["display"]["tile"] = discard;
		outputObj["display"]["tileCHI"] = midtile;
	}
}

void pung(int p, string discard, bool check) {
	players[p].hand.push_back(curtile);
	if(check)
		if(count(players[p].hand.begin(), players[p].hand.end(), curtile) < 3) {
			erroroutput(p, "WA");
			return;
		}
	for(int i = 0; i < 3; ++i)
		players[p].hand.erase(find(players[p].hand.begin(), players[p].hand.end(), curtile));
	players[p].melds.push_back({PUNG, curplayer, curtile});
	showntiles[curtile] += 2;
	curplayer = p;
	auto it = find(players[p].hand.begin(), players[p].hand.end(), discard);
	if(check)
		if(it == players[p].hand.end()) {
			erroroutput(p, "WA");
			return;
		}
	players[p].hand.erase(it);
	++showntiles[discard];
	walllast = players[(p + 1) % 4].tilewall.empty();
	curtile = discard;
	if(check) {
		for(int i = 0; i < 4; ++i)
			if(i != p)
				canHu[i] = checkmahjong(i, false, false, false);
		for(int i = 0; i < 4; ++i)
			outputObj["content"][to_string(i)] = "3 " + to_string(p) + " PENG " + discard;
		outputObj["display"]["action"] = "PENG";
		outputObj["display"]["player"] = p;
		outputObj["display"]["tile"] = discard;
	}
}

void meldedkong(int p, bool check) {
	players[p].hand.push_back(curtile);
	if(check)
		if(players[p].tilewall.empty() || walllast || count(players[p].hand.begin(), players[p].hand.end(), curtile) < 4) {
			erroroutput(p, "WA");
			return;
		}
	players[p].hand.erase(remove(players[p].hand.begin(), players[p].hand.end(), curtile), players[p].hand.end());
	players[p].melds.push_back({KONG, curplayer, curtile});
	showntiles[curtile] = 4;
	curplayer = p;
	drawaboutkong = true;
	isaboutkong = false;
	state = 0;
	if(check) {
		for(int i = 0; i < 4; ++i)
			outputObj["content"][to_string(i)] = "3 " + to_string(p) + " GANG";
		outputObj["display"]["action"] = "GANG";
		outputObj["display"]["player"] = p;
		outputObj["display"]["tile"] = curtile;
	}
}

void concealedkong(int p, string tile, bool check) {
	if(check)
		if(players[p].tilewall.empty() || walllast || count(players[p].hand.begin(), players[p].hand.end(), tile) < 4) {
			erroroutput(p, "WA");
			return;
		}
	players[p].hand.erase(remove(players[p].hand.begin(), players[p].hand.end(), tile), players[p].hand.end());
	players[p].melds.push_back({KONG, p, tile});
	showntiles[tile] = 4;
	drawaboutkong = true;
	isaboutkong = false;
	state = 0;
	if(check) {
		for(int i = 0; i < 4; ++i)
			outputObj["content"][to_string(i)] = "3 " + to_string(p) + " GANG";
		outputObj["display"]["action"] = "GANG";
		outputObj["display"]["player"] = p;
		outputObj["display"]["tile"] = tile;
	}
}

void promotedkong(int p, string tile, bool check) {
	auto tileit = find(players[p].hand.begin(), players[p].hand.end(), tile);
	auto meldit = find_if(players[p].melds.begin(), players[p].melds.end(),
		[&tile](const Meld &m) -> bool {
			return m.type == PUNG && m.tile == tile;
		}
	);
	if(check)
		if(players[p].tilewall.empty() || walllast || tileit == players[p].hand.end() || meldit == players[p].melds.end()) {
			erroroutput(p, "WA");
			return;
		}
	players[p].hand.erase(tileit);
	meldit->type = KONG;
	showntiles[tile] = 4;
	curtile = tile;
	drawaboutkong = true;
	isaboutkong = false;
	state = 3;
	if(check) {
		for(int i = 0; i < 4; ++i)
			if(i != p)
				canHu[i] = checkmahjong(i, false, true, false);
		for(int i = 0; i < 4; ++i)
			outputObj["content"][to_string(i)] = "3 " + to_string(p) + " BUGANG " + tile;
		outputObj["display"]["action"] = "BUGANG";
		outputObj["display"]["player"] = p;
		outputObj["display"]["tile"] = tile;
	}
}

void process(const Json::Value &log, bool check) {
	vector<string> response;
	string tmp;
	if(check)
		for(int i = 0; i < 4; ++i) {
			tmp = log[to_string(i)]["verdict"].asString();
			if(tmp != "OK") {
				erroroutput(i, tmp);
				return;
			}
			canHu[i] = -4;
		}
	switch(state) {
		case -1: // deal
			if(check) {
				for(int i = 0; i < 4; ++i) {
					tmp = log[to_string(i)]["response"].asString();
					if(tmp != "PASS") {
						erroroutput(i, "WA");
						return;
					}
				}
				outputObj["display"]["action"] = "DEAL";
				for(int i = 0; i < 4; ++i) {
					tmp = "1 0 0 0 0";
					for(auto &t : players[i].hand) {
						tmp += ' ';
						tmp += t;
						outputObj["display"]["hand"][i].append(t);
					}
					outputObj["content"][to_string(i)] = tmp;
				}
			}
			++state;
			break;
		case 0: // after deal / kong, about to draw
			if(check)
				for(int i = 0; i < 4; ++i) {
					tmp = log[to_string(i)]["response"].asString();
					if(tmp != "PASS") {
						erroroutput(i, "WA");
						return;
					}
				}
			draw(curplayer, check);
			break;
		case 1: // after draw
			if(check)
				for(int i = 0; i < 4; ++i)
					if(i != curplayer) {
						tmp = log[to_string(i)]["response"].asString();
						if(tmp != "PASS") {
							erroroutput(i, "WA");
							return;
						}
					}
			tmp = log[to_string(curplayer)]["response"].asString();
			split(response, tmp);
			if(response.size() == 1 && response[0] == "HU") {
				checkmahjong(curplayer, true, isaboutkong);
			} else if(response.size() == 2 && response[0] == "PLAY") {
				players[curplayer].hand.push_back(curtile);
				discard(curplayer, response[1], check);
			} else if(response.size() == 2 && response[0] == "GANG") {
				players[curplayer].hand.push_back(curtile);
				concealedkong(curplayer, response[1], check);
			} else if(response.size() == 2 && response[0] == "BUGANG") {
				players[curplayer].hand.push_back(curtile);
				promotedkong(curplayer, response[1], check);
			} else erroroutput(curplayer, "WA");
			break;
		case 2: // after play
			for(int o = 1; o < 4; ++o) { // check any hu
				int i = (o + curplayer) % 4;
				tmp = log[to_string(i)]["response"].asString();
				if(tmp == "HU") {
					checkmahjong(i, false, false);
					return;
				}
			}
			for(int o = 1; o < 4; ++o) { // check any pung / kong
				int i = (o + curplayer) % 4;
				tmp = log[to_string(i)]["response"].asString();
				split(response, tmp);
				if(response.size() == 1 && response[0] == "GANG") {
					meldedkong(i, check);
					return;
				} else if(response.size() == 2 && response[0] == "PENG") {
					pung(i, response[1], check);
					return;
				}
			}
			{ // check any chi
				int i = (1 + curplayer) % 4;
				tmp = log[to_string(i)]["response"].asString();
				split(response, tmp);
				if(response.size() == 3 && response[0] == "CHI") {
					chow(i, response[1], response[2], check);
					return;
				}
			}
			if(check)
				for(int i = 0; i < 4; ++i) {
					tmp = log[to_string(i)]["response"].asString();
					if(tmp != "PASS") {
						erroroutput(i, "WA");
						return;
					}
				}
			if(walllast) tieoutput(); // a tie
			else { // next player
				curplayer = (curplayer + 1) % 4;
				draw(curplayer, check);
			}
			break;
		case 3: // after promote kong
			for(int o = 0; o < 4; ++o) {
				int i = (o + curplayer) % 4;
				tmp = log[to_string(i)]["response"].asString();
				if(tmp == "HU" && i != curplayer) {
					checkmahjong(i, false, true);
					return;
				}
				else if(tmp != "PASS") {
					erroroutput(i, "WA");
					return;
				}
			}
			draw(curplayer, check);
			break;
	}
}

int main() {
	init();
	cin >> inputObj;
	initdata();
	outputObj["command"] = "request";
	if(inputObj["log"].empty()) initoutput();
	else {
		int n = inputObj["log"].size();
		state = -1;
		for(int i = 1; i < n; i += 2)
			process(inputObj["log"][i], i == n - 1);
	}
	for(int i = 0; i < 4; ++i) {
		outputObj["display"]["tileCnt"][i] = (int) players[i].tilewall.size();
		outputObj["display"]["canHu"][i] = canHu[i];
	}
	cout << outputObj;
}