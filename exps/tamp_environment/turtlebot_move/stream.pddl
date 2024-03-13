(define (stream rovers)
  (:stream sample-motion
    :inputs (?v ?q1 ?q2)
    :domain (and (Conf ?v ?q1) (Conf ?v ?q2))
    :outputs (?t)
    :certified (Motion ?v ?q1 ?t ?q2)
  )
  (:stream sample-motion-holding
    :inputs (?v ?q1 ?q2 ?o)
    :domain (and (Conf ?v ?q1) (Conf ?v ?q2) (Target ?o))
    :outputs (?t)
    :certified (MotionHolding ?v ?q1 ?t ?q2 ?o)
  )
  (:stream sample-ik
    :inputs (?v ?o ?p)
    :domain (and (Pose ?o ?p) (Rover ?v))
    :outputs (?q)
    :certified (and (Kin ?o ?p ?q) (Conf ?v ?q))
  )
)

  
