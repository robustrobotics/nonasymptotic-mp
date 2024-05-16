(define (stream kuka-tamp)
  (:stream plan-free-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :fluents (AtPose) ; AtGrasp
    :outputs (?t)
    ;:certified (and (Traj ?t) (FreeMotion ?q1 ?t ?q2))
    :certified (FreeMotion ?q1 ?t ?q2)
  )
  (:stream plan-holding-motion
    :inputs (?q1 ?q2 ?o ?g)
    :domain (and (Conf ?q1) (Conf ?q2) (Grasp ?o ?g))
    :fluents (AtPose)
    :outputs (?t)
    ;:certified (and (Traj ?t) (HoldingMotion ?q1 ?t ?q2 ?o ?g))
    :certified (HoldingMotion ?q1 ?t ?q2 ?o ?g)
  )
)