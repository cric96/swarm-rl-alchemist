package it.unibo.model

import it.unibo.model.Agent.Training
import it.unibo.model.DeepQLearner.computeNeighborhoodIndex
import it.unibo.model.network.torch._
import it.unibo.model.network.{DQN, DQNGNN, NeuralNetworkEncoding}
import it.unibo.util.LiveLogger
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

import java.text.SimpleDateFormat
import java.util.Date
import scala.util.Random

class DeepQLearner[State, Action](
    memory: ReplyBuffer[State, Action],
    actionSpace: Seq[Action],
    var epsilon: DecayReference[Double],
    gamma: Double,
    learningRate: Double,
    batchSize: Int = 32,
    val updateEach: Int = 100,
    val hiddenSize: Int = 32
)(implicit encoding: NeuralNetworkEncoding[State], random: Random)
    extends Agent[State, Action] {
  private var updates = 0
  private val targetNetwork = DQNGNN(2, hiddenSize, actionSpace.size)
  private val policyNetwork = DQNGNN(2, hiddenSize, actionSpace.size)
  private val targetPolicy = DeepQLearner.policyFromNetwork(policyNetwork, encoding, actionSpace)
  private val behaviouralPolicy = DeepQLearner.policyFromNetwork(policyNetwork, encoding, actionSpace)
  private val optimizer = optim.RMSprop(policyNetwork.parameters(), learningRate)
  val neigh = 5
  val True = torch.tensor(Seq(true).toPythonCopy)
  val optimal: State => Action = targetPolicy

  val behavioural: State => Action = state =>
    if (random.nextDouble() < epsilon) {
      random.shuffle(actionSpace).head
    } else behaviouralPolicy(state)

  override def record(state: State, action: Action, reward: Double, nextState: State): Unit =
    memory.insert(state, action, reward, nextState)

  override def improve(): Unit = if (this.mode == Training) {
    val memorySample = memory.sample(batchSize)
    if (memory.sample(batchSize).size == batchSize) {
      // val states = memorySample.map(_.state).toSeq.map(state => encoding.toSeq(state).toPythonCopy).toPythonCopy
      val states = memorySample.map(_.state).toSeq.map(state => encoding.toTensor(state))
      val indexes = states.map(state => computeNeighborhoodIndex((state.shape.bracketAccess(0) - 1).as[Int]))
      val graphsState = states.zip(indexes).map { case (x, index) => geometric.data.Data(x = x, edge_index = index) }
      val batchState = geometric.data.Batch.from_data_list(graphsState.toPythonCopy)
      val masks = states.map(state => torch.zeros(state.shape.bracketAccess(0) - 1, dtype = torch.bool))
      val maskWithMe = masks.map(mask => torch.cat(Seq(True, mask).toPythonCopy))
      val flattenMask = torch.cat(maskWithMe.toPythonCopy)
      val action = memorySample.map(_.action).toSeq.map(action => actionSpace.indexOf(action)).toPythonCopy
      val rewards = torch.tensor(memorySample.map(_.reward).toSeq.toPythonCopy)
      // val nextState = memorySample.map(_.nextState).toSeq.map(state => encoding.toSeq(state).toPythonCopy).toPythonCopy
      val nextState = memorySample.map(_.state).toSeq.map(state => encoding.toTensor(state))
      val graphsNextState =
        nextState.zip(indexes).map { case (x, index) => geometric.data.Data(x = x, edge_index = index) }
      val batchNextState = geometric.data.Batch.from_data_list(graphsNextState.toPythonCopy)
      val stateActionValue = policyNetwork(batchState.x, batchState.edge_index)
        .bracketAccess(flattenMask)
        .gather(1, torch.tensor(action).view(batchSize, 1))
      val nextStateValues = targetNetwork(batchNextState.x, batchNextState.edge_index)
        .bracketAccess(flattenMask)
        .max(1)
        .bracketAccess(0)
        .detach()
      /*val stateActionValue = policyNetwork(torch.tensor(states)).gather(1, torch.tensor(action).view(batchSize, 1))
      val nextStateValues = targetNetwork(torch.tensor(nextState)).max(1).bracketAccess(0).detach()*/
      val expectedValue = (nextStateValues * gamma) + rewards
      val criterion = nn.SmoothL1Loss()
      val loss = criterion(stateActionValue, expectedValue.unsqueeze(1))
      LiveLogger.logScalar("Loss", loss.item().as[Double], updates)
      optimizer.zero_grad()
      loss.backward()
      py"[param.grad.data.clamp_(-1, 1) for param in ${policyNetwork.parameters()}]"
      optimizer.step()
      updates += 1
      if (updates % updateEach == 0) {
        targetNetwork.load_state_dict(policyNetwork.state_dict())
      }
    }
  }

  def snapshot(episode: Int): Unit = {
    val timeMark = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date)
    torch.save(targetNetwork.state_dict(), s"data/network-$episode-$timeMark")
  }
}

object DeepQLearner {
  def policyFromNetworkSnapshot[S, A](
      path: String,
      hiddenSize: Int,
      encoding: NeuralNetworkEncoding[S],
      actionSpace: Seq[A]
  ): S => A = {
    val model = DQN(encoding.elements, hiddenSize, actionSpace.size)
    model.load_state_dict(torch.load(path))
    policyFromNetwork(model, encoding, actionSpace)
  }

  def policyFromNetwork[S, A](network: py.Dynamic, encoding: NeuralNetworkEncoding[S], actionSpace: Seq[A]): S => A = {
    state =>
      val netInput = encoding.toTensor(state)
      val neighborhoodIndexTorch = computeNeighborhoodIndex(netInput.shape.bracketAccess(0).as[Int] - 1)
      py.`with`(torch.no_grad()) { _ =>
        val data = network(netInput, neighborhoodIndexTorch)
        val actionIndex = data.bracketAccess(0).max(0).bracketAccess(1).item().as[Int]
        actionSpace(actionIndex)
      // actionSpace.head
      }
  }

  def computeNeighborhoodIndex(num: Int): py.Dynamic = {
    val neighborhoodIndex =
      List(List.fill(num)(0) ::: List.range(1, num + 1), (List.range(1, num + 1)) ::: List.fill(num)(0))
    val neighborhoodIndexPython = neighborhoodIndex.map(_.toPythonCopy).toPythonCopy
    torch.tensor(neighborhoodIndexPython)
  }
}
