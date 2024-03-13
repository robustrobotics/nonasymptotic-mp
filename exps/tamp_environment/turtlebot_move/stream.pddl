(define (stream rovers)
  (:stream sample-motion
    :inputs (?v ?q1 ?q2)
    :domain (and (Conf ?v ?q1) (Conf ?v ?q2))
    :outputs (?t)
    :certified (Motion ?v ?q1 ?t ?q2)
  )
  (:stream sample-ik
    :inputs (?o ?p)
    :domain (and (Pose ?o ?p))
    :outputs (?q)
    :certified (Kin ?o ?p ?q)
  )
)

  
