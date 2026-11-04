import { useState } from 'react'
import Board from './components/Board.jsx'
import Status from './components/Status.jsx'
import { bestMove } from './engine/random.js'
import { BOT, HUMAN, applyMove, createGame, getStatus, getTurn, isGameOver } from './engine/game.js'

function statusMessage(game) {
  const status = getStatus(game)
  if (status.kind === 'win') return status.winner === HUMAN ? 'You win' : 'Bot wins'
  return 'Your turn'
}

export default function App() {
  // Game state is immutable; each move produces a new state object.
  const [game, setGame] = useState(createGame)

  // The random bot is synchronous and instant, so the human's move and the
  // bot's reply land in a single state update.
  function handleMove(move) {
    if (getTurn(game) !== HUMAN || isGameOver(game)) return
    let next = applyMove(game, move)
    if (!next) return
    if (!isGameOver(next)) next = applyMove(next, bestMove(next))
    setGame(next)
  }

  const interactive = getTurn(game) === HUMAN && !isGameOver(game)

  return (
    <main className="app">
      <h1>Quoridor</h1>
      <Status message={statusMessage(game)} />
      <p className="walls-left">
        You: {game.wallsLeft[HUMAN]} walls · Bot: {game.wallsLeft[BOT]} walls
      </p>
      <Board game={game} interactive={interactive} onMove={handleMove} />
      <button className="new-game" onClick={() => setGame(createGame())}>
        New Game
      </button>
    </main>
  )
}
