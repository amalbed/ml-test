from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Molecules(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    smile = db.Column(db.String(64), nullable=False)

    def __repr__(self):
        return f"Molecule {self.smile}"

    def save(self):
        db.session.add(self)
        db.session.commit()

    @classmethod
    def get_by_id(cls, id):
        return cls.query.get_or_404(id)

    @classmethod
    def get_by_name(cls, smile):
        return cls.query.filter_by(smile=smile).first()

    def toDICT(self):

        cls_dict = {}
        cls_dict["_id"] = self.id
        cls_dict["smile"] = self.smile

        return cls_dict

    def toJSON(self):

        return self.toDICT()
