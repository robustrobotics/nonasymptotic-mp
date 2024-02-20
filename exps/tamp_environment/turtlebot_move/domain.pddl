(define (domain rovers)
  (:requirements :strips :equality)
  ; (:constants)
  (:predicates
    (Rover ?v)
    (Objective ?o)
    (Mode ?m)
    (Rock ?r)
    (Soil ?s)
    (Store ?s)
    (Lander ?l)
    (Type ?b ?t)

    (Conf ?v ?q)
    (Ray ?y)
    (Motion ?v ?q1 ?t ?q2)
    (ImageVisible ?v ?q ?y ?o)
    (ComVisible ?v ?q ?y ?l)
    (Above ?v ?q ?r)
    (CFreeRayConf ?y ?v ?q)

    (AtConf ?v ?q)
    (OnBoard ?c ?v)
    (Supports ?c ?m)
    (Calibrated ?c ?v)
    (HaveImage ?v ?o ?m)
    (ReceivedImage ?o ?m)
    (ReceivedAnalysis ?r)
    (Analyzed ?v ?r)
    (Free ?v ?s)
    (Full ?v ?s)

    (Blocked ?y)
  )

  (:action move
    :parameters (?v ?q1 ?t ?q2)
    :precondition (and (Motion ?v ?q1 ?t ?q2)
                       (AtConf ?v ?q1))
    :effect (and (AtConf ?v ?q2)
                 (not (AtConf ?v ?q1)))
  )

)