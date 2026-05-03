"""Quick replay analysis script: setup settlement pip quality."""
import sys, pickle
sys.path.insert(0, '.')
from replay_tools import ReplayData, ReplayEvent

def load(path):
    with open(path, 'rb') as f:
        class _U(pickle.Unpickler):
            def find_class(self, m, n):
                if n == 'ReplayData': return ReplayData
                if n == 'ReplayEvent': return ReplayEvent
                return super().find_class(m, n)
        return _U(f).load()

path = sys.argv[1] if len(sys.argv) > 1 else 'replays/self_play.pkl'
data = load(path)
board = data.states[0].board
topo  = board.topology

PIP_TABLE = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1}

def vertex_info(vid):
    pips, resources, details = 0, set(), []
    for hid in topo.vertex_hexes[vid]:
        h = board.hexes[hid]
        if h.hex_type.value != 'desert':
            pip = PIP_TABLE.get(h.token, 0)
            pips += pip
            resources.add(h.hex_type.value)
            details.append(f'{h.hex_type.value}({h.token}={pip}p)')
        else:
            details.append('desert(0p)')
    return pips, len(resources), ' + '.join(details)

print(f'Analyzing: {path}')
print(f'Winner: P{data.winner}\n')

print('=== SETUP SETTLEMENTS ===')
settle_count = 0
for e in data.events:
    if 'PLACE_SETTLEMENT' not in e.action_repr:
        continue
    vid = int(e.action_repr.split('v=')[1].rstrip(')'))
    pips, div, detail = vertex_info(vid)
    print(f'  P{e.player}: v={vid:3d}  pips={pips:2d}  div={div}  [{detail}]')
    settle_count += 1
    if settle_count >= 8:
        break

print()
print('=== ALL VERTICES RANKED BY PIPS (board reference) ===')
scored = [(vid, *vertex_info(vid)) for vid in range(topo.num_vertices)]
scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
for vid, pips, div, detail in scored[:20]:
    print(f'  v={vid:3d}  pips={pips:2d}  div={div}  [{detail}]')
