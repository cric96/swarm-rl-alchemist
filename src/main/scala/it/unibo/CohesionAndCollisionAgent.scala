package it.unibo

import it.unibo.AggregateComputingRLAgent.AgentResult
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.model.AgentAction.StandStill
import it.unibo.model.{AgentAction, State}
import it.unibo.scafi.space.Point3D

class CohesionAndCollisionAgent
    extends AggregateProgram
    with AggregateComputingRLAgent
    with FieldUtils
    with StandardSensors
    with BlockG
    with ScafiAlchemistSupport
    with CustomSpawn {
  def main(): AgentResult = {
    val distances = excludingSelf
      .reifyField(nbrVector())
      .toList
      .sortBy(_._2.distance(Point3D.Zero))
      .map(_._2)
      .map(point => (point.x, point.y))
    // .take(agentSpace)
    val state = State(distances, (0.0, 0.0))
    // if (distances.size == agentSpace) {
    AgentResult(state, getPolicy(state))
    // } else {
    //  AgentResult(state, AgentAction.West)
    // }
  }

  override def dropStandStill: Boolean = true

}
