(define (domain rovers)
  (:requirements :strips :equality)
  ; (:constants)
  (:predicates
    (Target ?o)
    (Conf ?v ?q)
    (Motion ?v ?q1 ?t ?q2)
    (HFree)
    (AtConf ?v ?q)
    (Pose ?o ?p)
    (AtPose ?o ?p)
    (Holding ?o)
    (Kin ?o ?p ?q)
    (Rover ?r)
    (MotionHolding ?v ?q1 ?t ?q2 ?o)
  )

  ;(:action move
  ;  :parameters (?v ?q1 ?t ?q2)
  ;  :precondition (and (Motion ?v ?q1 ?t ?q2)
  ;                     (AtConf ?v ?q1))
  ;  :effect (and (AtConf ?v ?q2)
  ;               (not (AtConf ?v ?q1)))
  ;)

  (:action pick
    :parameters (?v ?q1 ?t ?q2 ?p ?o)
    :precondition (and (Rover ?v)
                       (Motion ?v ?q1 ?t ?q2)
                       (AtConf ?v ?q1)
                       (AtPose ?o ?p)
                       (Kin ?o ?p ?q2)
                       (Target ?o)
                       (HFree))
    :effect (and (AtConf ?v ?q2)
                 (not (AtConf ?v ?q1))
                 (not (HFree))
                 (not (AtPose ?o ?p))
                 (Holding ?o)
            )
  )
  (:action place
    :parameters (?v ?q1 ?t ?q2 ?p ?o)
    :precondition (and (Rover ?v)
                       (MotionHolding ?v ?q1 ?t ?q2 ?o)
                       (AtConf ?v ?q1)
                       (Holding ?o)
                       (Target ?o)
                       (Pose ?o ?p)
                       (Kin ?o ?p ?q2)
                    )
    :effect (and (AtConf ?v ?q2)
                 (not (AtConf ?v ?q1))
                 (HFree)
                 (AtPose ?o ?p)
                 (not (Holding ?o))
            )
  )
)